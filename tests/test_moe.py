"""Behavioral tests for sparse replacement of a pretrained T5 FFN."""

import copy
import math
import unittest

import torch
from torch import nn

try:
    from moment.models.layers.moe_block import SparseMoE
except ModuleNotFoundError as error:
    if error.name != "moment.models.layers.moe_block":
        raise
    SparseMoE = None


class SparseMoETests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SparseMoE, "SparseMoE has not been implemented")
        torch.manual_seed(11)

    @staticmethod
    def make_weighted_moe():
        layer = SparseMoE(nn.Linear(2, 2, bias=False), 2, num_experts=4, top_k=2)
        with torch.no_grad():
            layer.router.weight.copy_(
                torch.tensor([[math.log(3), 0], [0, 0], [-math.log(3), 0], [-2 * math.log(3), 0]])
            )
            for expert, scale in zip(layer.experts, (2, 6, 10, 14)):
                expert.weight.copy_(scale * torch.eye(2))
        return layer

    def test_top_k_2_3_4_preserves_shape_and_dispatches_exactly_k_pairs(self):
        for top_k in (2, 3, 4):
            with self.subTest(top_k=top_k):
                layer = SparseMoE(nn.Linear(4, 4), 4, top_k=top_k)
                inputs = torch.randn(2, 3, 4)
                seen_rows = [0] * 8
                handles = []
                for index, expert in enumerate(layer.experts):
                    def record(_module, args, index=index):
                        seen_rows[index] += args[0].shape[0]
                    handles.append(expert.register_forward_pre_hook(record))
                output = layer(inputs)
                for handle in handles:
                    handle.remove()
                self.assertEqual(output.shape, inputs.shape)
                self.assertEqual(sum(seen_rows), 6 * top_k)
                self.assertEqual(layer.last_expert_counts.tolist(), seen_rows)
                self.assertTrue(torch.isfinite(output).all())

    def test_sparse_dispatch_uses_only_selected_tokens_and_normalizes_weights(self):
        layer = self.make_weighted_moe()
        seen_inputs = {}
        handles = []
        for index, expert in enumerate(layer.experts):
            handles.append(expert.register_forward_pre_hook(
                lambda _module, args, index=index: seen_inputs.update({index: args[0].detach().clone()})
            ))
        inputs = torch.tensor([[[1.0, 0], [-1.0, 0]]])
        output = layer(inputs)
        for handle in handles:
            handle.remove()
        # Positive token: (3/4)*2 + (1/4)*6 = 3.
        # Negative token: (3/4)*14 + (1/4)*10 = 13.
        torch.testing.assert_close(output, torch.tensor([[[3.0, 0], [-13.0, 0]]]))
        self.assertEqual(layer.last_expert_counts.tolist(), [1, 1, 1, 1])
        for index in (0, 1):
            torch.testing.assert_close(seen_inputs[index], inputs[0, :1])
        for index in (2, 3):
            torch.testing.assert_close(seen_inputs[index], inputs[0, 1:])

    def test_unselected_experts_are_never_called(self):
        layer = self.make_weighted_moe()
        called = []
        handles = [expert.register_forward_pre_hook(
            lambda _module, _args, index=index: called.append(index)
        ) for index, expert in enumerate(layer.experts)]
        layer(torch.tensor([[[1.0, 0], [2.0, 0]]]))
        for handle in handles:
            handle.remove()
        self.assertEqual(sorted(called), [0, 1])
        self.assertEqual(layer.last_expert_counts.tolist(), [2, 2, 0, 0])

    def test_router_selected_experts_and_inputs_receive_gradients(self):
        layer = self.make_weighted_moe()
        inputs = torch.tensor([[[1.0, 0.2], [2.0, -0.3]]], requires_grad=True)
        output = layer(inputs)
        loss = output.square().mean() + 0.01 * layer.aux_loss
        loss.backward()
        for gradient in (inputs.grad, layer.router.weight.grad,
                         layer.experts[0].weight.grad, layer.experts[1].weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNone(layer.experts[2].weight.grad)
        self.assertIsNone(layer.experts[3].weight.grad)

    def test_balance_loss_is_scalar_differentiable_and_penalizes_collapse(self):
        layer = self.make_weighted_moe()
        layer(torch.tensor([[[1.0, 0], [-1.0, 0]]]))
        balanced_loss = layer.aux_loss
        self.assertEqual(balanced_loss.shape, torch.Size([]))
        self.assertTrue(balanced_loss.requires_grad)
        # Every expert has the same assignment fraction, yielding E * (1/E) = 1.
        torch.testing.assert_close(balanced_loss, torch.tensor(1.0))
        layer(torch.tensor([[[1.0, 0], [1.0, 0]]]))
        self.assertGreater(layer.aux_loss.item(), balanced_loss.item())
        layer.aux_loss.backward()
        self.assertGreater(layer.router.weight.grad.abs().sum().item(), 0)

    def test_padding_has_zero_output_gradient_and_no_balance_contribution(self):
        layer = self.make_weighted_moe()
        reference = copy.deepcopy(layer)
        mask = torch.tensor([[True, False, True]])
        inputs = torch.tensor([[[1.0, 0], [-100.0, 70], [2.0, 0]]], requires_grad=True)
        layer.set_token_mask(mask)
        output = layer(inputs)
        reference_output = reference(inputs[:, [0, 2]])
        torch.testing.assert_close(output[:, [0, 2]], reference_output)
        self.assertTrue(torch.equal(output[:, 1], torch.zeros(1, 2)))
        torch.testing.assert_close(layer.aux_loss, reference.aux_loss)
        self.assertEqual(layer.last_expert_counts.tolist(), [2, 2, 0, 0])
        (output.sum() + layer.aux_loss).backward()
        self.assertTrue(torch.equal(inputs.grad[:, 1], torch.zeros(1, 2)))

    def test_no_valid_tokens_returns_finite_graph_connected_zero_without_expert_calls(self):
        layer = self.make_weighted_moe()
        inputs = torch.full((2, 3, 2), float("nan"), requires_grad=True)
        layer.set_token_mask(torch.zeros(2, 3, dtype=torch.bool))
        called = []
        handles = [expert.register_forward_pre_hook(
            lambda _module, _args: called.append(True)
        ) for expert in layer.experts]
        output = layer(inputs)
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(layer.aux_loss.item(), 0)
        self.assertTrue(layer.aux_loss.requires_grad)
        self.assertEqual(layer.last_expert_counts.tolist(), [0, 0, 0, 0])
        self.assertEqual(called, [])
        (output.sum() + layer.aux_loss).backward()
        self.assertTrue(torch.equal(inputs.grad, torch.zeros_like(inputs)))
        self.assertTrue(torch.isfinite(layer.router.weight.grad).all())

    def test_mask_can_be_cleared_between_forwards(self):
        layer = self.make_weighted_moe()
        inputs = torch.tensor([[[1.0, 0], [-1.0, 0]]])
        layer.set_token_mask(torch.tensor([[1, 0]]))
        self.assertEqual(layer(inputs)[0, 1].abs().sum().item(), 0)
        layer.set_token_mask(None)
        self.assertGreater(layer(inputs)[0, 1].abs().sum().item(), 0)
        self.assertEqual(layer.last_expert_counts.sum().item(), 4)

    def test_gated_t5_experts_preserve_pretrained_behavior_and_are_independent(self):
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

        config = T5Config(d_model=4, d_ff=9, dropout_rate=0.0, feed_forward_proj="gated-gelu")
        original = T5DenseGatedActDense(config).eval()
        layer = SparseMoE(original, 4, num_experts=4, top_k=2).eval()
        inputs = torch.randn(2, 3, 4)
        torch.testing.assert_close(layer(inputs), original(inputs))
        for expert in layer.experts:
            self.assertIsInstance(expert, T5DenseGatedActDense)
            self.assertNotEqual(expert.wi_0.weight.data_ptr(), original.wi_0.weight.data_ptr())
        first_weight = layer.experts[0].wi_0.weight.detach().clone()
        with torch.no_grad():
            layer.experts[1].wi_0.weight.add_(1)
        torch.testing.assert_close(layer.experts[0].wi_0.weight, first_weight)
        torch.testing.assert_close(original.wi_0.weight, first_weight)

    def test_cpu_bfloat16_autocast_keeps_float32_routing_finite(self):
        layer = self.make_weighted_moe()
        inputs = torch.tensor([[[10000.0, 0], [-10000.0, 0]]])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = layer(inputs)
        self.assertEqual(output.dtype, inputs.dtype)
        self.assertEqual(layer.aux_loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(layer.aux_loss))
        torch.testing.assert_close(output, torch.tensor([[[20000.0, 0], [-140000.0, 0]]]), rtol=0.01, atol=0)

    def test_float16_module_still_routes_in_float32(self):
        layer = self.make_weighted_moe().half()
        inputs = torch.tensor([[[1.0, 0], [-1.0, 0]]], dtype=torch.float16)
        output = layer(inputs)
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(layer.aux_loss.dtype, torch.float32)
        torch.testing.assert_close(output.float(), torch.tensor([[[3.0, 0], [-13.0, 0]]]), rtol=0.003, atol=0.003)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"top_k": 1}, {"top_k": 5}, {"top_k": 2.0},
                       {"num_experts": 1}, {"num_experts": 0}, {"num_experts": 2, "top_k": 3},
                       {"num_experts": 4.0}, {"d_model": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                options = {"d_model": 2, **kwargs}
                SparseMoE(nn.Linear(2, 2), **options)

    def test_input_and_mask_shape_mismatches_are_rejected(self):
        layer = self.make_weighted_moe()
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 3, 4))
        layer.set_token_mask(torch.ones(2, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 3, 2))


if __name__ == "__main__":
    unittest.main()
