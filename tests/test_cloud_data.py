"""Data tests run without importing MOMENT or Transformers."""
import importlib.util
from pathlib import Path
import unittest
import uuid

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pretraining/train_moe_vq.py"


class CloudDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("cloud_trainer", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_continuous_splits_precede_windows_and_fit_training_only(self):
        raw = np.arange(200, dtype=np.float32).reshape(100, 2)
        data = self.module.prepare_data(raw, layout="TC", seq_len=10, stride=3,
                                        train_fraction=.6, val_fraction=.2)
        self.assertEqual(data["split_bounds"], [0, 60, 80, 100])
        np.testing.assert_allclose(data["mean"], raw[:60].mean(axis=0))
        np.testing.assert_allclose(data["train"][0], ((raw[:10]-data["mean"])/data["std"]).T)
        self.assertEqual(data["train"].shape, (17, 2, 10))
        expected_val = ((raw[60:70]-data["mean"])/data["std"]).T
        np.testing.assert_allclose(data["validation"][0], expected_val)
        self.assertGreater(float(data["validation"].mean()), 1.)

    def test_ncl_split_keeps_channels_and_length(self):
        raw = np.arange(20*3*8, dtype=np.float32).reshape(20, 3, 8)
        data = self.module.prepare_data(raw, layout="NCL", seq_len=8, stride=8,
                                        train_fraction=.6, val_fraction=.2, standardize=False)
        np.testing.assert_array_equal(data["train"], raw[:12])
        np.testing.assert_array_equal(data["validation"], raw[12:16])
        np.testing.assert_array_equal(data["test"], raw[16:])

    def test_layout_mismatch_and_short_splits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "TC.*2"):
            self.module.prepare_data(np.ones((4, 2, 8)), layout="TC", seq_len=8, stride=8)
        with self.assertRaisesRegex(ValueError, "validation.*seq_len"):
            self.module.prepare_data(np.ones((30, 2)), layout="TC", seq_len=8, stride=8,
                                     train_fraction=.8, val_fraction=.1)
        with self.assertRaisesRegex(ValueError, "length"):
            self.module.prepare_data(np.ones((30, 2, 7)), layout="NCL", seq_len=8, stride=8)

    def test_nonfinite_inputs_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            raw = np.ones((100, 2), dtype=np.float32)
            raw[3, 1] = bad
            with self.assertRaisesRegex(ValueError, "finite"):
                self.module.prepare_data(raw, layout="TC", seq_len=8, stride=8)

    def test_csv_skips_date_and_preserves_numeric_channels(self):
        scratch = SCRIPT.parents[2] / "work"
        scratch.mkdir(exist_ok=True)
        path = scratch / f"cloud-data-{uuid.uuid4().hex}.csv"
        try:
            path.write_text("date,a,b\n2024-01-01,1,2\n2024-01-02,3,4\n", encoding="utf-8")
            np.testing.assert_array_equal(self.module.load_array(path, "TC"), [[1, 2], [3, 4]])
            path.write_text("date,a\n2024-01-01,1\n2024-01-02,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing|numeric"):
                self.module.load_array(path, "TC")
        finally:
            path.unlink(missing_ok=True)

    def test_synthetic_data_and_epoch_order_are_deterministic(self):
        first = self.module.synthetic_series(320, 2, 7)
        np.testing.assert_array_equal(first, self.module.synthetic_series(320, 2, 7))
        self.assertFalse(np.array_equal(first, self.module.synthetic_series(320, 2, 8)))
        order = self.module.epoch_order(11, 13, 2)
        np.testing.assert_array_equal(order, self.module.epoch_order(11, 13, 2))
        self.assertEqual(sorted(order.tolist()), list(range(11)))
        self.assertFalse(np.array_equal(order, self.module.epoch_order(11, 13, 3)))

    def test_invalid_mask_and_patch_configuration_is_rejected(self):
        good = {"model": {"seq_len": 32, "patch_len": 8, "patch_stride_len": 8,
                           "mask_ratio": .5}, "data": {"stride": 8},
                "training": {"batch_size": 2, "max_steps": 3}}
        self.module.validate_config(good)
        for field, value in (("patch_len", 32), ("mask_ratio", 0.), ("seq_len", 31)):
            bad = {**good, "model": {**good["model"], field: value}}
            with self.assertRaises(ValueError):
                self.module.validate_config(bad)


if __name__ == "__main__":
    unittest.main()
