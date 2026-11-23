"""Check independent Q/V LoRA and unpadding of Megatron sequences."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/megatron"))

from adapter import Float32Task, Layout, ShardedAdapter, split_packed_hidden


class MegatronAdapterTest(unittest.TestCase):
    def test_full_attention_q_and_v_leave_key_channels_unchanged(self):
        generator = torch.Generator().manual_seed(20261027)
        hidden = torch.randn(6, 5, generator=generator)
        shapes = (6, 2, 2)
        query = ShardedAdapter(5, Layout(4, shapes, (0,)), generator, 2, True)
        value = ShardedAdapter(5, Layout(4, shapes, (2,)), generator, 2, True)
        for adapter in (query, value):
            with torch.no_grad():
                adapter.lora_B.normal_(generator=generator)

        fused = hidden.new_zeros((6, 2 * sum(shapes)))
        fused = fused.index_add(-1, query.channels, query.expand(query.delta(hidden)))
        fused = fused.index_add(-1, value.channels, value.expand(value.delta(hidden)))
        q, k, v = fused.reshape(6, 2, -1).split(shapes, dim=-1)
        self.assertTrue(torch.equal(k, torch.zeros_like(k)))
        self.assertTrue(torch.equal(q.flatten(1), query.expand(query.delta(hidden))))
        self.assertTrue(torch.equal(v.flatten(1), value.expand(value.delta(hidden))))
        self.assertFalse(torch.equal(query.lora_A, value.lora_A))

    def test_gdn_shards_export_in_hf_q_k_v_order(self):
        layout = Layout(4, (2, 2, 3, 3, 1, 1), (0, 1, 2))
        local = layout.channels(2)
        self.assertEqual(local.tolist(), list(range(7)) + list(range(12, 19)))
        grouped = torch.arange(4 * 7).reshape(4, 7, 1)
        expected = torch.cat([part.flatten() for part in grouped.split((2, 2, 3), dim=1)])
        torch.testing.assert_close(layout.to_hf(grouped.flatten(0, 1)).flatten(), expected)

    def test_unpack_skips_padding_at_each_sequence_boundary(self):
        packed = torch.arange(8., requires_grad=True)
        hidden, mask = split_packed_hidden(packed[:, None, None], [0, 3, 8], [2, 3])
        self.assertEqual(hidden[:, :, 0].tolist(), [[0., 1., 0.], [3., 4., 5.]])
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])
        hidden.sum().backward()
        self.assertEqual(packed.grad.tolist(), [1., 1., 0., 1., 1., 1., 0., 0.])

    def test_mcore_bfloat16_does_not_round_task_weights(self):
        head = Float32Task()
        head.weight = torch.nn.Parameter(torch.tensor([1.000123], dtype=torch.float32))
        original = head.weight.detach().clone()
        head.bfloat16()
        torch.testing.assert_close(head.weight, original, rtol=0, atol=0)
        self.assertEqual(head.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
