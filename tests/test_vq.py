"""Input VQ behavior and persisted final-pass memory tests."""

import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import torch

try:
    from moment.models.layers.vq_vae import VQVAE
except ModuleNotFoundError as error:
    if error.name != "moment.models.layers.vq_vae":
        raise
    VQVAE = None


@contextmanager
def memory_directory():
    # Inherit workspace ACLs instead of tempfile's restrictive Windows mode.
    parent = Path(__file__).resolve().parent
    directory = parent / ("vq_memory_" + uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("vq_memory_"):
            raise RuntimeError("Refusing to remove a scratch path outside tests")
        shutil.rmtree(resolved)


class VQTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(VQVAE, "The input VQ module has not been implemented")
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def fixed_model(self):
        model = VQVAE(2, latent_dim=2, codebook_size=4, commitment_cost=0.25)
        with torch.no_grad():
            model.encoder.weight.copy_(torch.eye(2))
            model.encoder.bias.zero_()
            model.decoder.weight.copy_(torch.eye(2))
            model.decoder.bias.zero_()
            model.codebook.weight.copy_(torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.]]))
        return model

    def test_nearest_codes_and_hand_computed_losses(self):
        # Incorrect nearest lookup, loss weighting, or entropy must break this fixture.
        model = self.fixed_model()
        x = torch.tensor([[[[0.5, 0.], [1.5, 0.], [0., 1.5], [2., 1.5]]]])
        result = model(x)
        torch.testing.assert_close(result.indices, torch.tensor([[[0, 1, 2, 3]]]))
        torch.testing.assert_close(result.hidden_states, torch.tensor([[[[0., 0.], [2., 0.], [0., 2.], [2., 2.]]]]))
        self.assertAlmostEqual(result.codebook_loss.item(), 0.125)
        self.assertAlmostEqual(result.commitment_loss.item(), 0.125)
        self.assertAlmostEqual(result.loss.item(), 0.15625)
        self.assertAlmostEqual(result.perplexity.item(), 4.)

    def test_task_and_auxiliary_gradients_reach_correct_parameters(self):
        # Detaching the encoder/decoder path or codebook fitting must fail.
        model = self.fixed_model()
        x = torch.tensor([[[[0.5, 0.25], [1.5, 0.25]]]], requires_grad=True)
        result = model(x)
        result.hidden_states.sum().backward(retain_graph=True)
        for parameter in (model.encoder.weight, model.decoder.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(model.codebook.weight.grad)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        model.zero_grad(set_to_none=True)
        result.loss.backward()
        self.assertGreater(model.codebook.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.weight.grad.abs().sum().item(), 0)

    def test_hidden_tokens_bypass_exactly_and_do_not_affect_fitting(self):
        # Fitting hidden placeholders or replacing their values must fail.
        model = self.fixed_model()
        x = torch.tensor([[[[0.5, 0.], [99., -99.], [0., 1.5]]]], requires_grad=True)
        result = model(x, torch.tensor([[[True, False, True]]]))
        self.assertTrue(torch.equal(result.hidden_states[0, 0, 1], x[0, 0, 1]))
        torch.testing.assert_close(result.indices, torch.tensor([[[0, -1, 2]]]))
        self.assertAlmostEqual(result.loss.item(), 0.15625)
        torch.testing.assert_close(model.activation_counts, torch.tensor([1, 0, 1, 0]))
        result.hidden_states.sum().backward()
        torch.testing.assert_close(x.grad[0, 0, 1], torch.ones(2))

    def test_mask_broadcasts_over_batch_and_channels(self):
        model = self.fixed_model()
        x = torch.full((2, 3, 2, 2), 0.25)
        result = model(x, torch.tensor([True, False]))
        self.assertTrue((result.indices[:, :, 1] == -1).all())
        self.assertTrue(torch.equal(result.hidden_states[:, :, 1], x[:, :, 1]))
        self.assertEqual(model.activation_counts.sum().item(), 6)

    def test_all_invalid_is_finite_differentiable_and_leaves_counts_unchanged(self):
        # An empty mean or a zero derived from NaN placeholders must fail.
        model = self.fixed_model()
        x = torch.full((2, 1, 3, 2), float("nan"), requires_grad=True)
        model.begin_memory_collection()
        result = model(x, torch.zeros(2, 1, 3, dtype=torch.bool))
        for value in (result.loss, result.codebook_loss, result.commitment_loss, result.perplexity):
            self.assertEqual(value.item(), 0.)
            self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isnan(result.hidden_states).all())
        self.assertTrue((result.indices == -1).all())
        result.loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        self.assertEqual(model.activation_counts.sum().item(), 0)
        self.assertEqual(model.memory_counts.sum().item(), 0)

    def test_final_pass_bank_excludes_historical_and_evaluation_hits(self):
        # Historical diagnostics and normal validation must never populate the bank.
        model = self.fixed_model()
        model(torch.tensor([[0.25, 0.]]))
        self.assertEqual(model.memory_indices.numel(), 0)
        model.eval()
        before = model.activation_counts.clone()
        model(torch.tensor([[2., 2.]]))
        torch.testing.assert_close(model.activation_counts, before)
        self.assertEqual(model.memory_indices.numel(), 0)
        model.begin_memory_collection()
        model(torch.tensor([[2., 0.], [2., 0.], [0., 2.]]))
        model.end_memory_collection()
        torch.testing.assert_close(model.memory_counts, torch.tensor([0, 2, 1, 0]))
        torch.testing.assert_close(model.memory_indices, torch.tensor([1, 2]))
        torch.testing.assert_close(model.memory_bank, torch.tensor([[2., 0.], [0., 2.]]))
        model(torch.tensor([[2., 2.]]))
        torch.testing.assert_close(model.memory_counts, torch.tensor([0, 2, 1, 0]))
        model.begin_memory_collection()
        model(torch.tensor([[2., 2.]]))
        model.end_memory_collection()
        torch.testing.assert_close(model.memory_counts, torch.tensor([0, 0, 0, 1]))

    def test_state_dict_retains_fixed_size_diagnostics_and_bank(self):
        model = self.fixed_model()
        model(torch.tensor([[0.25, 0.]]))
        model.begin_memory_collection()
        model(torch.tensor([[2., 0.], [0., 2.]]))
        model.end_memory_collection()
        restored = VQVAE(2, latent_dim=2, codebook_size=4)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.memory_counts.shape, (4,))
        torch.testing.assert_close(restored.activation_counts, model.activation_counts)
        torch.testing.assert_close(restored.memory_counts, model.memory_counts)
        torch.testing.assert_close(restored.memory_bank, model.memory_bank)
        x = torch.tensor([[0.25, 0.], [0., 1.5]])
        torch.testing.assert_close(restored(x).hidden_states, model(x).hidden_states)

    def test_export_load_restores_codebook_and_bank(self):
        model = self.fixed_model().eval()
        model.begin_memory_collection()
        model(torch.tensor([[2., 0.], [0., 2.], [0., 2.]]))
        model.end_memory_collection()
        with memory_directory() as directory:
            path = Path(directory) / "memory.pt"
            model.export_memory(path)
            payload = torch.load(path, weights_only=True)
            self.assertIn("memory_bank", payload)
            torch.testing.assert_close(payload["memory_bank"], torch.tensor([[2., 0.], [0., 2.]]))
            restored = VQVAE(2, latent_dim=2, codebook_size=4).eval()
            restored.load_memory(path)
            torch.testing.assert_close(restored.codebook.weight, model.codebook.weight)
            torch.testing.assert_close(restored.memory_indices, torch.tensor([1, 2]))
            torch.testing.assert_close(restored.memory_counts, torch.tensor([0, 1, 2, 0]))
            torch.testing.assert_close(restored.memory_bank, model.memory_bank)
            restored(torch.tensor([[2., 2.]]))
            torch.testing.assert_close(restored.memory_counts, torch.tensor([0, 1, 2, 0]))

    def test_load_rejects_wrong_shapes_before_mutating_parameters(self):
        model = self.fixed_model()
        with memory_directory() as directory:
            path = Path(directory) / "memory.pt"
            model.export_memory(path)
            original = model.codebook.weight.detach().clone()
            for key, value in (("codebook", torch.zeros(3, 2)), ("memory_counts", torch.zeros(3, dtype=torch.long)),
                               ("memory_indices", torch.tensor([0]))):
                with self.subTest(key=key):
                    payload = torch.load(path, weights_only=True)
                    payload[key] = value
                    invalid_path = Path(directory) / f"invalid_{key}.pt"
                    torch.save(payload, invalid_path)
                    with self.assertRaises((ValueError, TypeError)):
                        model.load_memory(invalid_path)
                    torch.testing.assert_close(model.codebook.weight, original)
            with self.assertRaises(ValueError):
                VQVAE(3, latent_dim=2, codebook_size=4).load_memory(path)

    def test_load_rejects_bank_vectors_inconsistent_with_codebook_before_mutation(self):
        # A consumable bank must exactly match the exported codebook rows.
        model = self.fixed_model().eval()
        model.begin_memory_collection()
        model(torch.tensor([[2., 0.], [0., 2.]]))
        model.end_memory_collection()
        with memory_directory() as directory:
            path = directory / "memory.pt"
            model.export_memory(path)
            payload = torch.load(path, weights_only=True)
            payload["memory_bank"] = torch.tensor([[9., 0.], [0., 2.]])
            torch.save(payload, path)
            restored = VQVAE(2, latent_dim=2, codebook_size=4)
            original = {key: value.clone() for key, value in restored.state_dict().items()}
            with self.assertRaises(ValueError):
                restored.load_memory(path)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, original[key])

    def test_cpu_autocast_preserves_float32_losses(self):
        model = self.fixed_model()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model(torch.tensor([[0.5, 0.], [1.5, 0.]]))
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertEqual(result.codebook_loss.dtype, torch.float32)
        self.assertEqual(result.commitment_loss.dtype, torch.float32)
        torch.testing.assert_close(result.indices, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(result.loss))

    def test_rejects_invalid_configuration_and_masks(self):
        for kwargs in (dict(d_model=0), dict(d_model=2, latent_dim=0),
                       dict(d_model=2, codebook_size=0), dict(d_model=2, commitment_cost=-0.1),
                       dict(d_model=2, commitment_cost=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VQVAE(**kwargs)
        model = self.fixed_model()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 2), torch.ones(4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
