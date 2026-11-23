"""Preserve NAV2's independent rank-eight adapters in Megatron's fused layouts."""

import math
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Layout:
    groups: int
    widths: tuple[int, ...]
    parts: tuple[int, ...]

    @property
    def group_width(self):
        return sum(self.widths)

    def channels(self, groups):
        indices = []
        for part in self.parts:
            begin = sum(self.widths[:part])
            indices.extend(range(begin, begin + self.widths[part]))
        return (torch.arange(groups)[:, None] * self.group_width + torch.tensor(indices)).flatten()

    def to_hf(self, weight):
        weight = weight.reshape(self.groups, -1, weight.shape[-1])
        chunks = weight.split([self.widths[part] for part in self.parts], dim=1)
        return torch.cat([chunk.reshape(-1, weight.shape[-1]) for chunk in chunks])


class Float32Task(nn.Module):
    def _apply(self, function, recurse=True):
        def device_only(tensor):
            if not tensor.is_floating_point():
                return function(tensor)
            device = function(torch.empty(0, device=tensor.device, dtype=tensor.dtype)).device
            return tensor.to(device=device)

        return super()._apply(device_only, recurse=recurse)


class ShardedAdapter(Float32Task):
    def __init__(self, hidden_size, layout, generator, tp, sequence_parallel):
        super().__init__()
        if layout.groups % tp:
            raise ValueError("NAV2 adapter requires whole attention groups on each tensor-parallel rank")
        local_groups = layout.groups // tp
        self.layout = layout
        self.register_buffer("channels", layout.channels(local_groups), persistent=False)
        self.lora_A = nn.Parameter(torch.empty(8, hidden_size, dtype=torch.float32, device=generator.device))
        self.lora_B = nn.Parameter(torch.zeros(
            self.channels.numel(), 8, dtype=torch.float32, device=generator.device,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        if tp > 1:
            self.lora_B.tensor_model_parallel = True
            self.lora_B.partition_dim = 0
            self.lora_B.partition_stride = 1
            if sequence_parallel:
                self.lora_A.sequence_parallel = True

    def delta(self, normalized):
        return F.linear(normalized.float(), self.lora_A)

    def expand(self, projected):
        return F.linear(projected, self.lora_B) * 2


def split_packed_hidden(hidden, boundaries, lengths):
    if hidden.ndim != 3 or hidden.shape[1] != 1:
        raise ValueError(f"Expected a single flattened Megatron packed batch, got {hidden.shape}")
    if not lengths or len(boundaries) < len(lengths) + 1 or boundaries[0] != 0:
        raise ValueError("Packed sequence boundaries are missing")
    sequences = []
    for start, next_start, length in zip(boundaries, boundaries[1:], lengths):
        if length < 1 or start + length > min(hidden.shape[0], next_start):
            raise ValueError("Packed sequence lengths do not match the decoder hidden states")
        sequences.append(hidden[start:start + length, 0])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    mask = torch.arange(padded.size(1), device=hidden.device) < hidden.new_tensor(lengths, dtype=torch.long)[:, None]
    return padded, mask


def install(model):
    from megatron.core import parallel_state as mpu
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from megatron.core.tensor_parallel import (
        copy_to_tensor_model_parallel_region, gather_from_sequence_parallel_region,
    )

    from nav2_factorized_20261030 import FactHead

    config = model.config
    lm = model.language_model
    if config.linear_decoupled_in_proj or getattr(config, "mtp_num_layers", 0):
        raise ValueError("Use the Qwen3.5-4B Stage-1 fused GDN layout with no MTP auxiliary loss")
    tp = mpu.get_tensor_model_parallel_world_size()
    sequence_parallel = config.sequence_parallel
    task = Float32Task()
    task.adapters = nn.ModuleList()
    task.target_names = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261027)
        task.head = FactHead()
        torch.random.default_generator.manual_seed(20261112)
        task.input_projection = nn.Linear(config.hidden_size, 2560, bias=False)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20261027)
    group = mpu.get_tensor_model_parallel_group()

    def add(base, specs, local_sequence):
        if not isinstance(base, TELayerNormColumnParallelLinear):
            raise TypeError(f"Unexpected fusion module {type(base)}")
        adapters = []
        for name, layout in specs:
            if base.weight.shape[0] * tp != layout.groups * layout.group_width:
                raise ValueError(f"Unexpected fused projection width for {name}: {tuple(base.weight.shape)}")
            adapter = ShardedAdapter(config.hidden_size, layout, generator, tp, local_sequence)
            task.adapters.append(adapter)
            task.target_names.append(name)
            adapters.append(adapter)
        base.return_layernorm_output = True
        base.return_layernorm_output_gathered = False

        def apply_lora(module, inputs, output):
            (result, normalized), bias = output
            for adapter in adapters:
                projected = adapter.delta(normalized)
                if tp > 1 and local_sequence:
                    if normalized.shape[0] * tp != result.shape[0]:
                        raise ValueError("TE did not return a local sequence-parallel normalized shard")
                    projected = gather_from_sequence_parallel_region(projected, group=group)
                elif tp > 1:
                    if normalized.shape[0] != result.shape[0]:
                        raise ValueError("Expected the full input sequence after GDN's native gather")
                    projected = copy_to_tensor_model_parallel_region(projected, group=group)
                delta = adapter.expand(projected)
                result = result.index_add(-1, adapter.channels, delta.to(result.dtype))
            return result, bias

        base.register_forward_hook(apply_lora)

    for index, layer in enumerate(lm.decoder.layers):
        name = f"language_model.layers.{index}"
        attention = layer.self_attention
        if config.linear_attention_freq[index]:
            groups = config.linear_num_key_heads
            channels = config.linear_key_head_dim
            heads = config.linear_num_value_heads // groups
            value = config.linear_value_head_dim * heads
            layout = Layout(groups, (channels, channels, value, value, heads, heads), (0, 1, 2))
            # MCore-Bridge's GDN gathers the sequence before its input projection.
            add(attention.in_proj, [(f"{name}.linear_attn.in_proj_qkv", layout)], local_sequence=False)
        else:
            groups = config.num_query_groups
            channels = config.kv_channels
            query = channels * config.num_attention_heads // groups
            if config.attention_output_gate:
                query *= 2
            widths = (query, channels, channels)
            add(attention.linear_qkv, [
                (f"{name}.self_attn.q_proj", Layout(groups, widths, (0,))),
                (f"{name}.self_attn.v_proj", Layout(groups, widths, (2,))),
            ], local_sequence=sequence_parallel)
    if len(task.adapters) != 40:
        raise ValueError(f"Expected the original 40 Qwen3.5-4B LoRA adapters, got {len(task.adapters)}")

    def nav2_forward(lm, *, hidden_states, packed_seq_params, **kwargs):
        if packed_seq_params is None or not hasattr(packed_seq_params, "seq_lens"):
            raise ValueError("The official packing path must supply NAV2 sequence boundaries")
        if tp > 1 and sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=group,
            )
        spans = packed_seq_params.cu_seqlens_q.tolist()
        lengths = packed_seq_params.seq_lens.tolist()
        memory, attention_mask = split_packed_hidden(hidden_states, spans, lengths)
        head = lm.nav2
        with torch.autocast("cuda", enabled=False):
            heatmap, actions = head.head(head.input_projection(memory.float()), attention_mask)
        return torch.cat([heatmap.unsqueeze(-1), actions.flatten(-2)], dim=-1)

    lm.nav2 = task.to(device)
    lm._postprocess = MethodType(nav2_forward, lm)
    return task


def checkpoint(task, iteration):
    from megatron.core import parallel_state as mpu

    if mpu.get_data_parallel_rank() != 0:
        return None
    tp = mpu.get_tensor_model_parallel_world_size()
    group = mpu.get_tensor_model_parallel_group()
    state = {}
    for name, adapter in zip(task.target_names, task.adapters):
        local = adapter.lora_B.detach().contiguous()
        if tp > 1:
            shards = [torch.empty_like(local) for _ in range(tp)]
            torch.distributed.all_gather(shards, local, group=group)
            local = torch.cat(shards)
        if mpu.get_tensor_model_parallel_rank() == 0:
            state[name + ".lora_A"] = adapter.lora_A.detach().cpu()
            state[name + ".lora_B"] = adapter.layout.to_hf(local).cpu()
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None
    return dict(
        head={name: weight.detach().cpu() for name, weight in task.head.state_dict().items()},
        input_projection={name: weight.detach().cpu() for name, weight in task.input_projection.state_dict().items()},
        adapter=state, update=iteration, lora_rank=8, lora_alpha=16,
        targets=task.target_names, action_head="3x4", hidden_size=task.input_projection.in_features,
        observation_stride=1, training_framework="simplevln_ms_swift_megatron_sft",
    )
