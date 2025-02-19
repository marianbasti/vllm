import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any, Union, Dict, Set

from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.linear import (
    LinearMethodBase, ColumnParallelLinear, RowParallelLinear
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType, SimplePooler, PoolerHead
from vllm.sequence import SequenceOutput, IntermediateTensors, PoolerOutput, PoolingSequenceGroupOutput
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration, VllmModelForPooling
from vllm.config import VllmConfig, ModelConfig

class ModernBertRotaryEmbedding(RotaryEmbedding):
    """ModernBert-specific RoPE implementation."""
    def __init__(
        self,
        head_size: int,
        max_position: int = 8192,
        rotary_dim: Optional[int] = None,
        base: int = 10000,
        is_neox_style: bool = True,
        **kwargs
    ):
        # For ModernBert, rotary_dim is same as head_size if not specified
        rotary_dim = rotary_dim if rotary_dim is not None else head_size
        max_position_embeddings = max_position
        dtype = kwargs.get('dtype', torch.float32)
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(query=query, key=key, positions=positions)

class ModernBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', None)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.drop = nn.Dropout(p=getattr(config, 'hidden_dropout_prob', 0.0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.tok_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings

class ModernBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act = get_act_fn("gelu")
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ModernBertAttention(nn.Module):
    """Multi-head attention with RoPE."""
    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        # For ModernBert, Wqkv projects to 3 * hidden_size for Q,K,V
        qkv_proj_size = 3 * self.hidden_size  # This is fixed regardless of model size
            
        # Single QKV projection
        self.Wqkv = nn.Linear(self.hidden_size, qkv_proj_size, bias=False)
        
        # Output projection
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = ModernBertRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.head_size,  # Use full head dimension for rotation
            max_position=config.max_position_embeddings,
        )
        self.out_drop = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = hidden_states.size(0)
        qkv = self.Wqkv(hidden_states)  # [batch, seq, 3 * hidden]
        
        # Split into q, k, v each of size [batch, seq, hidden]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention: [batch, seq, num_heads, head_size]
        q = q.view(batch_size, -1, self.num_heads, self.head_size)
        k = k.view(batch_size, -1, self.num_heads, self.head_size)
        v = v.view(batch_size, -1, self.num_heads, self.head_size)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k, positions=positions)
        
        # Handle attention mask for flash attention
        attention_mask = None
        if hasattr(attn_metadata, 'attention_mask'):
            attention_mask = attn_metadata.attention_mask
        
        # Compute attention with flash attention if available
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=getattr(attn_metadata, 'is_causal', False)
        )
        
        # Reshape and project back
        attn_output = attn_output.reshape(batch_size, -1, self.hidden_size)
        attn_output = self.Wo(attn_output)
        attn_output = self.out_drop(attn_output)
        
        return attn_output, None  # Return None as bias since we don't use it

class ModernBertMLP(nn.Module):
    """MLP with GeGLU activation."""
    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Set proper intermediate size based on model size
        if self.hidden_size == 768:  # base model
            self.intermediate_size = 2304  # For base model
            self.gating_size = 1152     # Half of intermediate size
        elif self.hidden_size == 1024:  # large model
            self.intermediate_size = 5248  # For large model
            self.gating_size = 2624      # Half of intermediate size
        else:
            raise ValueError(f"Unsupported hidden size: {self.hidden_size}")
        
        # Wi projects to intermediate_size for both gate and transform parts
        self.Wi = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act = get_act_fn("gelu")
        self.drop = nn.Dropout(p=getattr(config, 'hidden_dropout_prob', 0.0))
        # Wo projects from gating_size back to hidden_size
        self.Wo = nn.Linear(self.gating_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Wi(x)  # [batch, seq_len, intermediate_size]
        # Split into gate and transform tensors
        gate, transform = x.chunk(2, dim=-1)  # Each has size [..., gating_size]
        x = self.act(gate) * transform  # GeGLU activation
        x = self.drop(x)
        x = self.Wo(x)  # Project back to hidden size
        return x

class ModernBertEncoderLayer(nn.Module):
    # Map checkpoint names to model names
    name_mapping = {
        'attention.self.query.weight': 'attn.Wqkv.weight',  # Map all Q/K/V to single QKV projection
        'attention.output.dense.weight': 'attn.Wo.weight',
        'attention.output.LayerNorm': 'attn_norm',
        'intermediate.dense.weight': 'mlp.Wi.weight',
        'output.dense.weight': 'mlp.Wo.weight',
        'output.LayerNorm': 'mlp_norm',
    }

    def __init__(self, config, layer_id: int = 0):
        super().__init__()
        # First layer has Identity for attn_norm
        self.attn_norm = (nn.Identity() if layer_id == 0 
                         else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False))
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.mlp = ModernBertMLP(config)

    def load_weights(self, weights: Any, layer_prefix: str) -> set:
        """Load weights specific to this encoder layer"""
        loaded_weights = set()
        qkv_tensors = {}
        
        for name, param in weights:
            if not name.startswith(layer_prefix):
                continue
            
            # Remove the layer prefix to get local parameter name
            local_name = name[len(layer_prefix):].lstrip('.')
            
            # Special handling for QKV weights
            if any(qkv in local_name for qkv in ['query.weight', 'key.weight', 'value.weight']):
                qkv_type = local_name.split('.')[2]  # Extract query/key/value
                qkv_tensors[qkv_type] = param
                if len(qkv_tensors) == 3:
                    # Concatenate in proper order for ModernBERT QKV
                    qkv_weight = torch.cat([
                        qkv_tensors['query'],
                        qkv_tensors['key'],
                        qkv_tensors['value']
                    ], dim=0)
                    
                    self.attn.Wqkv.weight.data.copy_(qkv_weight)
                    loaded_weights.add(f"{layer_prefix}.attention.self.query.weight")
                    loaded_weights.add(f"{layer_prefix}.attention.self.key.weight")
                    loaded_weights.add(f"{layer_prefix}.attention.self.value.weight")
                continue

            # Special handling for MLP weights - they need proper chunking for GeGLU
            if 'intermediate.dense.weight' in local_name:
                # Wi weight needs to be properly chunked for gate and transform
                self.mlp.Wi.weight.data.copy_(param)
                loaded_weights.add(name)
                continue
                
            if 'output.dense.weight' in local_name:
                # Wo weight takes the already chunked size as input
                self.mlp.Wo.weight.data.copy_(param)
                loaded_weights.add(name)
                continue

            # Handle regular weights using name mapping
            mapped_name = local_name
            for old_name, new_name in self.name_mapping.items():
                if old_name in mapped_name:
                    mapped_name = mapped_name.replace(old_name, new_name)
                    break

            # Try to find and set the parameter
            for module_name, module in self.named_modules():
                param_name = mapped_name.replace(module_name + '.', '') if module_name else mapped_name
                if hasattr(module, param_name):
                    getattr(module, param_name).data.copy_(param)
                    loaded_weights.add(name)
                    break

        return loaded_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed_hidden_states = self.attn_norm(hidden_states)
        attention_output, _ = self.attn(  # Unpack tuple, ignore bias
            normed_hidden_states,
            kv_cache,
            attn_metadata,
            positions=positions
        )
        hidden_states = hidden_states + attention_output
        
        # Pre-norm MLP
        mlp_normed = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_normed)  # MLP already handles bias internally
        hidden_states = hidden_states + mlp_output
        
        return hidden_states

class ModernBertModel(nn.Module):
    """ModernBert base model."""
    # Map checkpoint names to model names
    name_mapping = {
        'embeddings.word_embeddings': 'embeddings.tok_embeddings',
        'embeddings.position_embeddings': None,  # We don't use these
        'embeddings.token_type_embeddings': None,  # We don't use these
        'embeddings.LayerNorm': 'embeddings.norm',
        'encoder.layer': 'layers',  # Base prefix for transformer layers
        'pooler': None,  # We don't use the original pooler
        'encoder.LayerNorm': 'final_norm',
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        
        self.embeddings = ModernBertEmbeddings(self.config)
        
        # Initialize layers with proper layer_id
        self.layers = nn.ModuleList([
            ModernBertEncoderLayer(self.config, layer_id=i)
            for i in range(self.config.num_hidden_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps, elementwise_affine=False)

    def load_weights(self, weights: Any) -> Optional[set]:
        loaded_weights = set()
        layer_weights = {}  # Group weights by layer
        
        # First pass - group weights by layer and handle non-layer weights
        for name, param in weights:
            if 'encoder.layer.' in name:
                # Extract layer number and store weight
                layer_idx = int(name.split('encoder.layer.')[1].split('.')[0])
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = []
                layer_weights[layer_idx].append((name, param))
                continue
            
            # Handle non-layer weights using name mapping
            mapped_name = name
            for old_name, new_name in self.name_mapping.items():
                if new_name is not None and old_name in mapped_name:
                    mapped_name = mapped_name.replace(old_name, new_name)
                    break
            
            if mapped_name is None:
                continue  # Skip weights we don't use
                
            # Find the parameter in our model
            found = False
            for param_name, _ in self.named_parameters():
                if mapped_name.endswith(param_name):
                    # Get the parameter and copy the data
                    param_parts = param_name.split('.')
                    curr = self
                    for part in param_parts[:-1]:
                        curr = getattr(curr, part)
                    param_attr = getattr(curr, param_parts[-1])
                    param_attr.data.copy_(param)
                    loaded_weights.add(name)
                    found = True
                    break
            
            if not found and hasattr(self, mapped_name):
                getattr(self, mapped_name).data.copy_(param)
                loaded_weights.add(name)
        
        # Second pass - load layer weights
        for layer_idx, weights in layer_weights.items():
            if layer_idx >= len(self.layers):
                continue
            layer_loaded = self.layers[layer_idx].load_weights(
                weights, f'encoder.layer.{layer_idx}'
            )
            if layer_loaded:
                loaded_weights.update(layer_loaded)
        
        return loaded_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Use provided embeddings if available, otherwise compute them
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        
        # Handle intermediate tensors if provided
        if intermediate_tensors is not None:
            return intermediate_tensors
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                kv_cache=kv_caches[i] if kv_caches else None,
                attn_metadata=attn_metadata,
                positions=positions
            )
            
        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states

class ModernBertForMaskedLM(VllmModelForTextGeneration, nn.Module):
    # Map checkpoint names to model names
    name_mapping = {
        'bert': 'model',
        'cls.predictions.transform': 'head',
        'cls.predictions.decoder': 'decoder',
        'cls.predictions.bias': 'decoder.bias',
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        VllmModelForTextGeneration.__init__(self, vllm_config=vllm_config)
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = ModernBertModel(vllm_config=vllm_config, prefix=prefix)
        
        # Match checkpoint's prediction head architecture
        self.head = ModernBertPredictionHead(self.config)
        
        # Initialize decoder with bias only, weight will be tied with embeddings
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)
        # Tie weights with embedding layer
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def load_weights(self, weights: Any) -> Optional[set]:
        """Load weights from an iterator of (name, tensor) pairs."""
        loaded_weights = set()
        bert_weights = []  # Group BERT weights
        head_weights = []  # Group head weights
        decoder_weights = []  # Group decoder weights
        
        # First pass - group weights by component
        for name, param in weights:
            if name.startswith('cls.predictions.transform'):
                head_weights.append((name, param))
            elif name.startswith('cls.predictions'):
                decoder_weights.append((name, param))
            elif name.startswith('bert'):
                bert_weights.append((name, param))
            else:
                continue
                
        # Load BERT weights
        if bert_weights:
            loaded = self.model.load_weights(bert_weights)
            if loaded:
                loaded_weights.update(loaded)
                
        # Load head weights
        params_dict = dict(self.head.named_parameters())
        for name, param in head_weights:
            # Map weight names from checkpoint to our model
            if 'dense.weight' in name:
                if 'dense.weight' in params_dict:
                    params_dict['dense.weight'].data.copy_(param)
                    loaded_weights.add(name)

        # Load decoder weights (bias only since weights are tied)
        if hasattr(self.decoder, 'bias'):
            for name, param in decoder_weights:
                if 'bias' in name:
                    self.decoder.bias.data.copy_(param)
                    loaded_weights.add(name)

        return loaded_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[SequenceOutput, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
            
        hidden_states = self.head(hidden_states)
        logits = self.decoder(hidden_states)
        
        # Return last token's logits as output_token and None for parent_seq_id and logprobs
        return SequenceOutput(
            parent_seq_id=None,  
            output_token=logits[:, -1, :],  # Get logits of last token
            logprobs=None  # We don't compute logprobs for MLM
        )

class ModernBertForMaskedLMForEmbedding(VllmModelForTextGeneration, nn.Module):
    """ModernBert for masked language modeling with embedding layer."""
    is_text_generation_model = True
    name_mapping = {
        'bert': 'model',
        'embeddings.word_embeddings': 'embeddings',
        'cls.predictions.decoder': 'decoder',
        'cls.predictions.bias': 'decoder.bias',
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs: Any,
    ):
        VllmModelForTextGeneration.__init__(self, vllm_config=vllm_config)
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        
        # For embedding task, we only need the embeddings layer
        self.embeddings = VocabParallelEmbedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            params_dtype=vllm_config.model_config.dtype,
        )
        # Add bias for compatibility with the decoder in MLM task
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)
        # Tie the weights
        self.decoder.weight = self.embeddings.weight

    def load_weights(self, weights_iter) -> set:
        """Handle weight loading for embedding-only model"""
        loaded_weights = set()
        for name, param in weights_iter:
            mapped_name = name
            for old_name, new_name in self.name_mapping.items():
                if new_name is not None:
                    mapped_name = mapped_name.replace(old_name, new_name)

            if mapped_name == 'decoder.bias':
                self.decoder.bias.data.copy_(param)
                loaded_weights.add(name)
            elif mapped_name == 'embeddings.weight':
                self.embeddings.weight.data.copy_(param)
                loaded_weights.add(name)
                # Since weights are tied, we also mark the decoder weight as loaded
                loaded_weights.add('cls.predictions.decoder.weight')
        return loaded_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> SequenceOutput:
        if inputs_embeds is not None:
            embeddings = inputs_embeds
        else:
            embeddings = self.embeddings(input_ids)
            
        # Return embeddings in SequenceOutput format
        return SequenceOutput(
            parent_seq_id=None,
            output_token=embeddings[:, -1, :],  # Get last token embedding
            logprobs=None  # Embedding-only model doesn't compute logprobs
        )

class ModernBertForPooling(VllmModelForTextGeneration, VllmModelForPooling):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        VllmModelForTextGeneration.__init__(self, vllm_config=vllm_config)
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = ModernBertModel(vllm_config=vllm_config, prefix=prefix)
        
        # Initialize pooler
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, "Pooler config must be provided for pooling model"
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,  # Default for embedding
            normalize=True,
            softmax=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pool the hidden states using the configured pooler."""
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Any) -> Optional[set]:
        """Load weights from an iterator of (name, tensor) pairs."""
        return self.model.load_weights(weights)