import unittest
import os, sys

import torch

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)

from util import json_cfg
from util.json_cfg import schema_path
from Trainer import Trainer


class TestTrainer(unittest.TestCase):
    def test_init_trainer(self):
        cfg_path = os.path.join(test_dir, "test_cfg", "test_cfg.json")
        cfg = json_cfg.load_cfg(cfg_path)
        schema = json_cfg.load_cfg(schema_path)
        self.assertTrue(json_cfg.validate_cfg(cfg=cfg, schema=schema))
        trainer = Trainer(cfg, verbose=False)
        self.assertEqual(trainer.name, cfg["name"])
        self.assertEqual(trainer.method, cfg["method"])
        self.assertEqual(trainer.epochs, cfg["epochs"])


if __name__ == "__main":
    unittest.main()
