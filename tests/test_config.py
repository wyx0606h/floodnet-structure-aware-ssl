from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from floodnet_ssl.config import ConfigError, load_yaml_config


VALID_CONFIG = """
experiment:
  run_id: test
  output_dir: runs/test
  seed: 1
  kind: overfit4
data:
  data_root: ${FLOODNET_TEST_ROOT}
  manifest: splits/test.csv
  crop_size: 512
model:
  name: segformer_b0
  num_labels: 10
  pretrained: true
  pretrained_model_name_or_path: nvidia/mit-b0
  local_files_only: true
training:
  epochs: 2
  batch_size: 1
  gradient_accumulation_steps: 1
  optimizer: adamw
  learning_rate: 0.001
evaluation:
  tile_size: 512
  stride: 384
overfit_gate:
  maximum_final_to_initial_loss_ratio: 0.2
  minimum_train_miou10: 0.9
"""


class ConfigTest(unittest.TestCase):
    def test_loads_and_expands_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(VALID_CONFIG, encoding="utf-8")
            previous = os.environ.get("FLOODNET_TEST_ROOT")
            os.environ["FLOODNET_TEST_ROOT"] = str(Path(temporary) / "data")
            try:
                config = load_yaml_config(path)
            finally:
                if previous is None:
                    os.environ.pop("FLOODNET_TEST_ROOT", None)
                else:
                    os.environ["FLOODNET_TEST_ROOT"] = previous
            self.assertEqual(
                str(Path(temporary) / "data"), config["data"]["data_root"]
            )

    def test_rejects_noncanonical_crop_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                VALID_CONFIG.replace("crop_size: 512", "crop_size: 256"),
                encoding="utf-8",
            )
            os.environ["FLOODNET_TEST_ROOT"] = temporary
            with self.assertRaises(ConfigError):
                load_yaml_config(path)


    def test_new_protocol_configs_load(self) -> None:
        sup = load_yaml_config(Path("configs/segformer_b0_sup398.yaml"))
        full = load_yaml_config(Path("configs/segformer_b0_full1445.yaml"))
        self.assertEqual("sup398", sup["dataset"]["protocol"])
        self.assertEqual("full1445", full["dataset"]["protocol"])
        comparable_keys = ["model", "loss", "training", "evaluation"]
        for key in comparable_keys:
            self.assertEqual(sup[key], full[key])


    def test_new_protocol_training_policy_values(self) -> None:
        sup = load_yaml_config(Path("configs/segformer_b0_sup398.yaml"))
        training = sup["training"]
        data = sup["data"]
        model = sup["model"]
        self.assertEqual(40000, training["max_iterations"])
        self.assertEqual(2000, training["val_interval"])
        self.assertEqual(2, training["batch_size"])
        self.assertEqual(4, training["gradient_accumulation_steps"])
        self.assertEqual("poly", training["scheduler"])
        self.assertEqual(1000, training["warmup_iterations"])
        self.assertEqual(1.0, training["poly_power"])
        self.assertEqual(1.0, training["gradient_clip_norm"])
        self.assertTrue(data["drop_last"])
        self.assertEqual(0.0, data["class_aware_probability"])
        self.assertFalse(model["local_files_only"])


if __name__ == "__main__":
    unittest.main()
