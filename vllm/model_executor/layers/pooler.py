# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing_extensions import assert_never

from vllm.config import PoolerConfig
from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput, SequenceOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3
    MEAN = 4


class SimplePooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    @staticmethod
    def from_pooling_type(
        pooling_type: PoolingType,
        *,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> "SimplePooler":
        if pooling_type == PoolingType.LAST:
            assert step_tag_id is None and returned_token_ids is None
            return LastPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.ALL:
            assert step_tag_id is None and returned_token_ids is None
            return AllPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.CLS:
            assert step_tag_id is None and returned_token_ids is None
            return CLSPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.MEAN:
            assert step_tag_id is None and returned_token_ids is None
            return MeanPool(normalize=normalize, softmax=softmax)
        if pooling_type == PoolingType.STEP:
            return StepPool(normalize=normalize,
                            softmax=softmax,
                            step_tag_id=step_tag_id,
                            returned_token_ids=returned_token_ids)

        assert_never(pooling_type)

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.head = PoolerHead(normalize=normalize, softmax=softmax)

    def get_prompt_lens(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        device = pooling_metadata.device if hasattr(pooling_metadata, 'device') else torch.device('cuda')
        return PoolingTensors.from_pooling_metadata(pooling_metadata, device).prompt_lens

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def build_output(self, data: torch.Tensor) -> PoolingSequenceGroupOutput:
        return PoolingSequenceGroupOutput(data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data)
        pooled_outputs = [self.build_output(data) for data in pooled_data]
        return PoolerOutput(outputs=pooled_outputs)


class CLSPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        data = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.data

        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return data[first_token_flat_indices]


class LastPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        
        # Handle different input types for hidden_states
        if isinstance(hidden_states, torch.Tensor):
            data = hidden_states
        elif isinstance(hidden_states, SequenceOutput):
            # For single SequenceOutput object, get output_token directly
            device = pooling_metadata.device if hasattr(pooling_metadata, 'device') else torch.device('cuda')
            data = hidden_states.output_token.to(device)
        elif isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
            # For sequence/list of SequenceOutput objects
            device = pooling_metadata.device if hasattr(pooling_metadata, 'device') else torch.device('cuda')
            if isinstance(hidden_states[0], SequenceOutput):
                data = torch.stack([s.output_token for s in hidden_states]).to(device)
            else:
                data = torch.stack(hidden_states).to(device)
        else:
            raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")
        
        last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
        return data[last_token_flat_indices]


class AllPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        data = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.data

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len in prompt_lens:
            pooled_data.append(data[offset:offset + prompt_len])
            offset += prompt_len

        return pooled_data


class MeanPool(SimplePooler):

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        data = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.data

        cumsum = torch.cumsum(data, dim=0)
        start_indices = torch.cat([
            torch.tensor([0], device=data.device),
            torch.cumsum(prompt_lens[:-1], dim=0)
        ])
        end_indices = torch.cumsum(prompt_lens, dim=0)
        return (cumsum[end_indices - 1] - cumsum[start_indices] +
                data[start_indices]) / prompt_lens.unsqueeze(1)


class StepPool(SimplePooler):

    def __init__(
        self,
        *,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ):
        super().__init__(normalize=normalize, softmax=softmax)

        self.step_tag_id = step_tag_id
        self.returned_token_ids = returned_token_ids

    def extract_states(
        self,
        hidden_states: Union[torch.Tensor, Any],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        data = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.data

        returned_token_ids = self.returned_token_ids
        if returned_token_ids is not None and len(returned_token_ids) > 0:
            data = data[:, returned_token_ids]

        step_tag_id = self.step_tag_id

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len, seq_data_i in zip(prompt_lens,
                                          pooling_metadata.seq_data.values()):
            pooled_data_i = data[offset:offset + prompt_len]
            if step_tag_id is not None:
                token_ids = torch.tensor(seq_data_i.prompt_token_ids)
                pooled_data_i = pooled_data_i[token_ids == step_tag_id]

            offset += prompt_len
            pooled_data.append(pooled_data_i)

        return pooled_data


class PoolerHead(nn.Module):

    def __init__(self, *, normalize: bool, softmax: bool) -> None:
        super().__init__()

        self.normalize = normalize
        self.softmax = softmax

    def forward(self, pooled_data: Union[list[torch.Tensor], torch.Tensor]):
        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    F.normalize(data, p=2, dim=1) for data in pooled_data
                ]
            else:
                pooled_data = F.normalize(pooled_data, p=2, dim=1)

        if self.softmax:
            if isinstance(pooled_data, list):
                pooled_data = [F.softmax(data, dim=-1) for data in pooled_data]
            else:
                pooled_data = F.softmax(pooled_data, dim=-1)

        return pooled_data


class Pooler(nn.Module):

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> SimplePooler:
        return SimplePooler.from_pooling_type(
            pooling_type=PoolingType[pooler_config.pooling_type]
            if pooler_config.pooling_type is not None else pooling_type,
            normalize=pooler_config.normalize
            if pooler_config.normalize is not None else normalize,
            softmax=pooler_config.softmax
            if pooler_config.softmax is not None else softmax,
            step_tag_id=pooler_config.step_tag_id
            if pooler_config.step_tag_id is not None else step_tag_id,
            returned_token_ids=pooler_config.returned_token_ids
            if pooler_config.returned_token_ids is not None else
            returned_token_ids,
        )


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = \
            get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset:offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        pooled_outputs = [PoolingSequenceGroupOutput(data) for data in scores]
        return PoolerOutput(outputs=pooled_outputs)
