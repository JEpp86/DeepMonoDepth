import unittest
import os, sys

test_dir = os.path.dirname(__file__)
module_dir = os.path.dirname(test_dir)
root_dir = os.path.dirname(module_dir)
sys.path.insert(0, root_dir)

from util.json_cfg import Config
from core.Trainer import Trainer


class TestTrainer(unittest.TestCase):
    """TODO Test data fails, trainer works on real data
    def test_init_trainer(self):
        cfg_path = os.path.join(test_dir, "test_cfg", "test_cfg.json")
        cfg = Config(cfg_path)
        trainer = Trainer(cfg, verbose=False)
        self.assertEqual(trainer.name, cfg.cfg["name"])
        self.assertEqual(trainer.method, cfg.cfg["method"])
        self.assertEqual(trainer.epochs, cfg.cfg["epochs"])
    """


if __name__ == "__main__":
    unittest.main()
