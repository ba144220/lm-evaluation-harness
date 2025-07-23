from typing import Optional, List
import torch.nn as nn
import torch

from transformers import Qwen3ForCausalLM, Qwen3Config, Qwen3Model
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

class SkipLayersDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_skipped = False
        self.config = config
    def set_is_skipped(self, is_skipped: bool) -> None:
        self.is_skipped = is_skipped
    def get_is_skipped(self) -> bool:
        return self.is_skipped
    def get_layer_idx(self) -> int:
        return self.self_attn.layer_idx
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_value: Optional[Cache] = None,
        *args,
        **kwargs
    ):
        
        if self.is_skipped:

            # This part is a workaround to fake the kv cache. 
            if past_key_value is not None:
                k_shape = (hidden_states.shape[0], self.config.num_key_value_heads, hidden_states.shape[1], self.config.hidden_size // self.config.num_attention_heads)

                key_states = torch.randn(k_shape, device=hidden_states.device, dtype=hidden_states.dtype)
                value_states = torch.randn(k_shape, device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value.update(key_states, value_states, self.get_layer_idx())

            outputs = (hidden_states, )
            if output_attentions:
                outputs += (None, )
            return outputs

        
        return super().forward(
            hidden_states, 
            output_attentions=output_attentions, 
            past_key_value=past_key_value,
            *args, 
            **kwargs 
        )

class SkipLayersModel(Qwen3Model):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.layers = nn.ModuleList(
            [SkipLayersDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def get_skip_layers(self) -> List[int]:
        skip_layers = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.get_is_skipped(): # type: ignore
                skip_layers.append(layer_idx)
        return skip_layers
    def set_skip_layers(self, skip_layers: List[int]) -> None:
        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            layer.set_is_skipped(layer_idx in skip_layers) # type: ignore


class SkipLayersForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = SkipLayersModel(config)

    def get_skip_layers(self) -> List[int]:
        return self.model.get_skip_layers()
    def set_skip_layers(self, skip_layers: List[int]) -> None:
        self.model.set_skip_layers(skip_layers)
