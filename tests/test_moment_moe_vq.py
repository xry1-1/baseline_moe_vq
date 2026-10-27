import unittest
import warnings

import torch

from moment.models.moment import MOMENT


def tiny_config(**overrides):
    config = dict(
        task_name="pre-training", seq_len=32, patch_len=4, patch_stride_len=4,
        transformer_backbone="t5-small", transformer_type="encoder_only",
        transformer_config=dict(d_model=16, d_ff=32, d_kv=8, num_layers=1,
                                num_heads=2, vocab_size=32, dropout_rate=0.0,
                                feed_forward_proj="gated-gelu"),
        randomly_initialize_backbone=True, d_model=16, device="cpu",
        dropout=0.0, orth_gain=None, add_positional_embedding=False,
        mask_ratio=0.5, debug=False, enable_gradient_checkpointing=False,
        use_moe=True, moe_num_experts=4, moe_top_k=2, use_vq=True,
        vq_latent_dim=8, vq_codebook_size=16,
    )
    config.update(overrides)
    return config


class MomentIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_offline_input_vq_moe_backward(self):
        model = MOMENT(tiny_config())
        x = torch.randn(2, 2, 32)
        valid = torch.ones(2, 32, dtype=torch.long)
        visible = valid.clone()
        visible[:, 8:16] = 0
        recorded = {}
        hook = model.encoder.register_forward_pre_hook(
            lambda module, args, kwargs: recorded.update(inputs=kwargs["inputs_embeds"].detach()),
            with_kwargs=True,
        )
        output = model(x, input_mask=valid, mask=visible)
        hook.remove()
        self.assertEqual(output.reconstruction.shape, x.shape)
        self.assertEqual(output.vq_indices.shape, (2, 2, 8))
        self.assertTrue((output.vq_indices[:, :, 2:4] == -1).all())
        expected_mask = model.patch_embedding.mask_embedding.detach().reshape(-1)
        torch.testing.assert_close(recorded["inputs"][:, 2:4], expected_mask.expand(4, 2, 16))
        loss = (output.reconstruction - x).square().mean() + output.aux_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.vq_vae.codebook.weight.grad.abs().sum().item(), 0)
        router = model.encoder.block[0].layer[-1].DenseReluDense.router
        self.assertGreater(router.weight.grad.abs().sum().item(), 0)

    def test_missing_mask_and_nondefault_patch_length(self):
        model = MOMENT(tiny_config())
        result = model(torch.randn(2, 1, 32))
        self.assertEqual(result.pretrain_mask.shape, (2, 32))
        self.assertEqual(result.vq_indices.shape, (2, 1, 8))
        self.assertTrue(torch.isfinite(result.reconstruction).all())

    def test_common_paths_and_bank_state_roundtrip(self):
        config = tiny_config()
        model = MOMENT(config).eval()
        x = torch.randn(2, 1, 32)
        valid = torch.ones(2, 32, dtype=torch.long)
        valid[0, -4:] = 0
        model.vq_vae.begin_memory_collection()
        with torch.no_grad():
            reconstructed = model.reconstruct(x, input_mask=valid)
        model.vq_vae.end_memory_collection()
        self.assertTrue((reconstructed.vq_indices[0, :, -1] == -1).all())
        self.assertGreater(model.vq_vae.memory_indices.numel(), 0)
        restored = MOMENT(config).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored.reconstruct(x, valid).reconstruction,
                                       reconstructed.reconstruction)
            self.assertEqual(restored.embed(x, valid).embeddings.shape, (2, 16))
            self.assertEqual(restored.short_forecast(x, valid, forecast_horizon=4).forecast.shape,
                             (2, 1, 4))
        torch.testing.assert_close(restored.vq_vae.memory_counts, model.vq_vae.memory_counts)
        forecast_model = MOMENT(tiny_config(task_name="long-horizon-forecasting", forecast_horizon=5))
        self.assertEqual(forecast_model(x, input_mask=valid).forecast.shape, (2, 1, 5))

    def test_disabled_features_and_all_padding(self):
        plain = MOMENT(tiny_config(use_moe=False, use_vq=False))
        output = plain(torch.randn(2, 1, 32))
        self.assertEqual(float(output.aux_loss), 0.0)
        model = MOMENT(tiny_config())
        output = model(torch.zeros(2, 1, 32), input_mask=torch.zeros(2, 32, dtype=torch.long))
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertTrue((output.vq_indices == -1).all())
        self.assertTrue(torch.isfinite(output.reconstruction).all())

    def test_checkpointing_is_disabled_for_moe(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MOMENT(tiny_config(enable_gradient_checkpointing=True))
        self.assertFalse(model.encoder.gradient_checkpointing)
        self.assertTrue(any("checkpoint" in str(w.message).lower() for w in caught))

    def test_dense_checkpoint_initializes_all_experts(self):
        plain = MOMENT(tiny_config(use_moe=False, use_vq=False))
        enhanced = MOMENT(tiny_config())
        transferred = enhanced.load_pretrained_state_dict(plain.state_dict())
        self.assertGreater(len(transferred), 0)
        dense = plain.encoder.block[0].layer[-1].DenseReluDense
        for expert in enhanced.encoder.block[0].layer[-1].DenseReluDense.experts:
            for name, value in dense.state_dict().items():
                torch.testing.assert_close(expert.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
