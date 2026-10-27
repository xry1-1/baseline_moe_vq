"""Legacy trainers and latest-checkpoint lookup remain usable with enhancements."""

import __future__
import ast
import os
import shutil
import subprocess
import unittest
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from moment.models.base import BaseModel
from moment.utils.optims import LinearWarmupCosineLRScheduler
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists


PROJECT = Path(__file__).resolve().parents[1]


@contextmanager
def regression_directory():
    parent = (PROJECT / "work").resolve()
    directory = parent / ("legacy-regression-" + uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("legacy-regression-"):
            raise RuntimeError("Refusing to remove a fixture outside the project work directory")
        shutil.rmtree(resolved)


def load_pretraining_class():
    """Execute the unchanged class ASTs without W&B or dataset import side effects.

    All training, validation, optimizer, scheduler and checkpoint methods below
    come directly from the production classes. Only their external logger and
    model/data construction are supplied by the test fixture.
    """
    namespace = {
        "__name__": "legacy_training_under_test",
        "os": os,
        "subprocess": subprocess,
        "warnings": warnings,
        "deepcopy": deepcopy,
        "np": np,
        "torch": torch,
        "nn": nn,
        "optim": torch.optim,
        "tqdm": tqdm,
        "AlertLevel": SimpleNamespace(INFO="INFO"),
        "MetricsStore": MetricsStore,
        "dtype_map": dtype_map,
        "make_dir_if_not_exists": make_dir_if_not_exists,
        "LinearWarmupCosineLRScheduler": LinearWarmupCosineLRScheduler,
    }
    for relative_path, class_name in (
        ("moment/tasks/base.py", "Tasks"),
        ("moment/tasks/pretrain.py", "Pretraining"),
    ):
        path = PROJECT / relative_path
        source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(selected) != 1:
            raise AssertionError(f"Expected one production {class_name} class")
        module = ast.Module(body=selected, type_ignores=[])
        exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["Pretraining"]


class ReconstructionOnlyModel(nn.Module):
    """A real differentiable model with the pre-existing GPT4TS output contract."""

    def __init__(self, optional_fields="absent"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.optional_fields = optional_fields
        self.forward_calls = 0

    def forward(self, x_enc, input_mask, mask=None):
        self.forward_calls += 1
        output = SimpleNamespace(
            reconstruction=x_enc * self.scale,
            pretrain_mask=torch.zeros_like(input_mask),
        )
        if self.optional_fields == "none":
            output.aux_loss = output.vq_loss = output.moe_aux_loss = None
        elif self.optional_fields == "values":
            output.aux_loss = 2 * self.scale.square()
            output.vq_loss = self.scale * 0 + 0.2
            output.moe_aux_loss = self.scale * 0 + 0.3
        return output


class RecordingLogger:
    def __init__(self, name):
        self.name = name
        self.records = []

    def log(self, record):
        self.records.append(record)

    def alert(self, **kwargs):
        pass


class LegacyPretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production_class = load_pretraining_class()

    def make_task(self, directory, optional_fields="absent"):
        # Bypass dataset/model construction; execute the actual production loops.
        task = self.production_class.__new__(self.production_class)
        nn.Module.__init__(task)
        task.args = SimpleNamespace(
            checkpoint_path=str(directory), model_name="GPT4TS", torch_dtype="float32",
            use_amp=False, set_input_mask=True, debug=False, max_opt_steps=1,
            max_epoch=3, max_norm=10.0, log_interval=100, checkpoint_interval=100,
            optimizer_name="SGD", init_lr=0.1, momentum=0.0, weight_decay=0.0,
            lr_decay_rate=0.9, warmup_lr=0.0, warmup_steps=0, min_lr=0.0,
        )
        task.device = torch.device("cpu")
        task.model = ReconstructionOnlyModel(optional_fields)
        task.logger = RecordingLogger(optional_fields)
        batch = SimpleNamespace(timeseries=torch.ones(1, 1, 4), input_mask=torch.ones(1, 4))
        task.train_dataloader = [batch, batch]
        task.val_dataloader = [batch]
        task.criterion = nn.MSELoss(reduction="none")
        return task

    def test_validation_accepts_outputs_with_missing_or_none_auxiliary_fields(self):
        with regression_directory() as directory:
            for optional_fields in ("absent", "none"):
                with self.subTest(optional_fields=optional_fields):
                    task = self.make_task(directory, optional_fields)
                    average, losses, (trues, predictions, masks) = task.validation(
                        task.val_dataloader, return_preds=True
                    )
                    self.assertAlmostEqual(average, 0.25)
                    np.testing.assert_allclose(losses, [0.25])
                    np.testing.assert_allclose(predictions, trues * 0.5)
                    np.testing.assert_array_equal(masks, np.zeros((1, 4)))
                    self.assertTrue(task.model.training)

    def test_train_reconstruction_only_model_updates_once_and_saves_final_checkpoint(self):
        with regression_directory() as directory:
            for optional_fields in ("absent", "none"):
                with self.subTest(optional_fields=optional_fields):
                    task = self.make_task(directory, optional_fields)
                    result = task.train()
                    self.assertIs(result, task.model)
                    self.assertEqual(task.model.forward_calls, 1)
                    self.assertAlmostEqual(task.model.scale.item(), 0.6, places=6)
                    self.assertAlmostEqual(task.logger.records[0]["step_train_loss"], 0.25)
                    checkpoint = torch.load(
                        directory / optional_fields / "GPT4TS_checkpoint_1.pth",
                        map_location="cpu", weights_only=False,
                    )
                    self.assertEqual(checkpoint["opt_steps"], 1)
                    torch.testing.assert_close(checkpoint["model_state_dict"]["scale"], task.model.scale)

    def test_present_auxiliary_losses_still_affect_validation_training_and_logging(self):
        with regression_directory() as directory:
            task = self.make_task(directory, "values")
            self.assertAlmostEqual(task.validation(task.val_dataloader), 0.75)
            task.model.forward_calls = 0
            task.train()
            self.assertEqual(task.model.forward_calls, 1)
            self.assertAlmostEqual(task.model.scale.item(), 0.4, places=6)
            record = task.logger.records[0]
            self.assertAlmostEqual(record["step_train_loss"], 0.75)
            self.assertAlmostEqual(record["reconstruction_loss"], 0.25)
            self.assertAlmostEqual(record["vq_loss"], 0.2)
            self.assertAlmostEqual(record["moe_aux_loss"], 0.3)


class LatestCheckpointTests(unittest.TestCase):
    def write_checkpoint(self, directory, filename, step):
        checkpoint = {
            "step": step,
            "model_state_dict": {"weight": torch.tensor([float(step)])},
            "config": SimpleNamespace(model_name="MOMENT"),
            "rng": np.random.RandomState(7).get_state(),
        }
        torch.save(checkpoint, directory / filename)

    def test_latest_lookup_ignores_bank_unrelated_files_and_checkpoint_named_directories(self):
        with regression_directory() as directory:
            run = directory / "run"
            run.mkdir()
            self.write_checkpoint(run, "moment_checkpoint_2.pth", 2)
            self.write_checkpoint(run, "MOMENT_checkpoint_11.pth", 11)
            torch.save({"bank": torch.tensor([0])}, run / "codebook_memory.pt")
            (run / "training.jsonl").write_text("{}\n", encoding="utf-8")
            (run / "moment_checkpoint_99.pth").mkdir()
            for filename in ("other_checkpoint_300.pth", "moment_checkpoint_400.pth.bak",
                             "moment_checkpoint_bad.pth"):
                (run / filename).write_text("unrelated", encoding="utf-8")
            checkpoint = BaseModel.load_pretrained_weights("run", checkpoints_dir=str(directory))
            self.assertEqual(checkpoint["step"], 11)
            self.assertEqual(checkpoint["model_state_dict"]["weight"].device.type, "cpu")
            self.assertEqual(checkpoint["config"].model_name, "MOMENT")

    def test_explicit_step_loads_lower_or_uppercase_and_trusted_metadata(self):
        with regression_directory() as directory:
            run = directory / "run"
            run.mkdir()
            self.write_checkpoint(run, "moment_checkpoint_2.pth", 2)
            self.write_checkpoint(run, "MOMENT_checkpoint_11.pth", 11)
            for step in (2, 11):
                with self.subTest(step=step):
                    checkpoint = BaseModel.load_pretrained_weights(
                        "run", opt_steps=step, checkpoints_dir=str(directory)
                    )
                    self.assertEqual(checkpoint["step"], step)
                    self.assertEqual(checkpoint["model_state_dict"]["weight"].device.type, "cpu")
                    self.assertEqual(checkpoint["config"].model_name, "MOMENT")
                    np.testing.assert_array_equal(checkpoint["rng"][1], np.random.RandomState(7).get_state()[1])

    def test_no_checkpoint_files_raises_clear_file_not_found(self):
        with regression_directory() as directory:
            run = directory / "run"
            run.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "[Cc]heckpoint"):
                BaseModel.load_pretrained_weights("run", checkpoints_dir=str(directory))
            torch.save({"bank": torch.zeros(1)}, run / "codebook_memory.pt")
            with self.assertRaisesRegex(FileNotFoundError, "[Cc]heckpoint"):
                BaseModel.load_pretrained_weights("run", checkpoints_dir=str(directory))

    def test_missing_explicit_step_raises_file_not_found(self):
        with regression_directory() as directory:
            run = directory / "run"
            run.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint_3"):
                BaseModel.load_pretrained_weights("run", opt_steps=3, checkpoints_dir=str(directory))


if __name__ == "__main__":
    unittest.main()
