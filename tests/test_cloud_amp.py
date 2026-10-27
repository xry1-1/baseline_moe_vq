"""Exercise the trainer's update boundary with real GradScaler and optimizer."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pretraining/train_moe_vq.py"


class CloudAMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("cloud_amp_trainer", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_scaler_overflow_skips_without_clipping_then_next_batch_updates(self):
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.)
        # A gradient hook makes overflow reproducible without numerical flakiness.
        hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        loss = parameter.square().sum()
        result = self.module.optimizer_update(loss, [parameter], optimizer, scaler, 1.)
        self.assertFalse(result["optimizer_updated"])
        self.assertEqual(parameter.item(), 2.)
        self.assertEqual(scaler.get_scale(), 64.)
        self.assertIsNone(result["grad_norm"])
        # Clipping Inf would overwrite this with NaN; the overflow path must bypass it.
        self.assertTrue(torch.isinf(parameter.grad).all())
        output = io.StringIO()
        self.module.log_event(output, "overflow", **result)
        self.assertIsNone(json.loads(output.getvalue())["grad_norm"])

        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        result = self.module.optimizer_update(parameter.square().sum(), [parameter], optimizer, scaler, 1.)
        self.assertTrue(result["optimizer_updated"])
        self.assertAlmostEqual(parameter.item(), 1.9, places=6)
        self.assertAlmostEqual(result["grad_norm"], 4., places=6)

    def test_nonfinite_gradient_without_scaling_fails_before_optimizer_update(self):
        for context in (False, True):
            with self.subTest(cpu_bfloat16_autocast=context):
                parameter = torch.nn.Parameter(torch.tensor([2.]))
                optimizer = torch.optim.SGD([parameter], lr=.1)
                scaler = torch.amp.GradScaler("cpu", enabled=False)
                parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=context):
                    loss = parameter.square().sum()
                with self.assertRaisesRegex((RuntimeError, FloatingPointError), "non-finite|Nonfinite"):
                    self.module.optimizer_update(loss, [parameter], optimizer, scaler, 1.)
                self.assertEqual(parameter.item(), 2.)


class CloudAblationCLITests(unittest.TestCase):
    def test_baseline_and_moe_only_complete_without_bank_and_reject_memory_only_early(self):
        import yaml

        scratch = SCRIPT.parents[2]/"work"/f"cloud-ablation-{uuid.uuid4().hex}"
        scratch.mkdir(parents=True)
        try:
            spec = importlib.util.spec_from_file_location("cloud_ablation_trainer", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = yaml.safe_load((SCRIPT.parents[2]/"configs/pretraining/moe_vq.yaml").read_text(encoding="utf-8"))
            config = module.smoke_config(config)
            config["model"]["use_vq"] = False
            config_path = scratch/"config.yaml"
            for use_moe in (False, True):
                with self.subTest(use_moe=use_moe):
                    config["model"]["use_moe"] = use_moe
                    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    output = scratch/str(use_moe)
                    result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config_path),
                        "--synthetic", "--device", "cpu", "--max-steps", "1", "--output-dir", str(output)],
                        capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    checkpoint = torch.load(output/"checkpoint_final.pt", weights_only=False, map_location="cpu")
                    self.assertEqual(checkpoint["step"], 1)
                    self.assertFalse(checkpoint["bank_ready"])
                    self.assertFalse((output/"codebook_memory.pt").exists())
                    forbidden_output = scratch/f"memory-only-{use_moe}"
                    result = subprocess.run([sys.executable, str(SCRIPT), "--resume", str(output/"checkpoint_final.pt"),
                        "--build-memory-only", "--device", "cpu", "--output-dir", str(forbidden_output)],
                        capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("use_vq", result.stderr)
                    self.assertFalse(forbidden_output.exists(), "VQ requirement must fail before creating output")
        finally:
            # Only remove the dedicated UUID fixture directory within the project workspace.
            if scratch.resolve().parent != (SCRIPT.parents[2]/"work").resolve():
                raise AssertionError("Unsafe test cleanup path")
            shutil.rmtree(scratch)


if __name__ == "__main__":
    unittest.main()
