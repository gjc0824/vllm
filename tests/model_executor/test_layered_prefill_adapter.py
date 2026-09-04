# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.layered_prefill import (
    StandardDecoderLayeredPrefillAdapter,
)
from vllm.sequence import IntermediateTensors


class _DecoderLayer(nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = increment

    def forward(self, positions, hidden_states, residual):
        del positions
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        residual = residual + hidden_states
        return hidden_states + self.increment, residual


class _FinalNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class _RequiredStateLayer(nn.Module):
    def forward(self, positions, hidden_states, residual, required_state):
        return hidden_states, residual, required_state


class _OptionalStateLayer(nn.Module):
    def forward(self, positions, hidden_states, residual, optional_state=None):
        return hidden_states, residual


class _ExtraKwargsLayer(nn.Module):
    def forward(self, positions, hidden_states, residual, **kwargs):
        del kwargs
        return hidden_states, residual


class _DecoderBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.start_layer = 0
        self.end_layer = 2
        self.layers = nn.ModuleList([_DecoderLayer(1), _DecoderLayer(2)])
        self.norm = _FinalNorm()

    @staticmethod
    def embed_input_ids(input_ids):
        return input_ids.float().unsqueeze(-1)

    @staticmethod
    def make_empty_intermediate_tensors(batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(batch_size, 1, dtype=dtype, device=device),
                "residual": torch.zeros(batch_size, 1, dtype=dtype, device=device),
            }
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return self.norm(hidden_states, residual)[0]


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DecoderBackbone()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)


def test_standard_adapter_matches_full_forward_without_model_hook():
    model = _CausalLM()
    adapter = StandardDecoderLayeredPrefillAdapter(model)
    input_ids = torch.tensor([2, 5])
    positions = torch.arange(2)

    first = adapter.forward(
        input_ids=input_ids,
        positions=positions,
        layer_start=0,
        layer_end=1,
    )
    second = adapter.forward(
        input_ids=input_ids,
        positions=positions,
        layer_start=1,
        layer_end=2,
        frontier=(first.hidden_states, first.residual),
    )

    assert not hasattr(model, "forward_layered_prefill")
    assert first.is_final_layer is False
    assert second.is_final_layer is True
    torch.testing.assert_close(second.hidden_states, model(input_ids, positions))


def test_standard_adapter_uses_normal_pp_intermediate_schema():
    model = _CausalLM()
    adapter = StandardDecoderLayeredPrefillAdapter(model)
    hidden_states, residual = adapter.make_transport_frontier(
        3, torch.float32, torch.device("cpu")
    )

    assert hidden_states.shape == (3, 1)
    assert residual is not None and residual.shape == (3, 1)

    output = adapter.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        layer_start=0,
        layer_end=1,
    )
    intermediate_tensors = adapter.to_intermediate_tensors(output)
    assert set(intermediate_tensors.tensors) == {"hidden_states", "residual"}


def test_standard_adapter_accepts_ignored_extra_kwargs():
    model = _CausalLM()
    model.model.layers[0] = _ExtraKwargsLayer()
    adapter = StandardDecoderLayeredPrefillAdapter(model)

    assert isinstance(adapter, StandardDecoderLayeredPrefillAdapter)


def test_standard_adapter_inspects_original_contract_through_wrapper():
    model = _CausalLM()
    layer = model.model.layers[0]
    original_forward = layer.forward

    def wrapped_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    layer.forward = wrapped_forward
    adapter = StandardDecoderLayeredPrefillAdapter(model)

    assert isinstance(adapter, StandardDecoderLayeredPrefillAdapter)


@pytest.mark.parametrize(
    "special_layer",
    [_RequiredStateLayer(), _OptionalStateLayer()],
)
def test_standard_adapter_rejects_nonstandard_layer_state(special_layer):
    model = _CausalLM()
    model.model.layers[0] = special_layer

    with pytest.raises(TypeError, match="standard decoder layer contract"):
        StandardDecoderLayeredPrefillAdapter(model)
