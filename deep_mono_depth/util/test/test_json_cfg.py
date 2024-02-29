import unittest
import os, sys

test_dir = os.path.dirname(__file__)
module_dir = os.path.dirname(test_dir)
root_dir = os.path.dirname(module_dir)
proj_dir = os.path.dirname(root_dir)
sys.path.insert(0, proj_dir)
from json_cfg import Config


class TestCfg(unittest.TestCase):
    """
    def test_read(self):
        cfg = json_cfg.Config(os.path.join(test_dir, "test_cfg.json"))
        self.assertTrue(cfg["name"] == "Test")
        self.assertTrue(cfg["pi"] == 3.14159)
        self.assertTrue(cfg["something"]["thing_one"] == "one")
        self.assertTrue(len(cfg["some_array"]) == 4)

    def test_validate(self):
        schema = json_cfg.load_cfg(os.path.join(test_dir, "test_cfg.schema.json"))
        cfg = json_cfg.load_cfg(os.path.join(test_dir, "test_cfg.json"))
        self.assertTrue(json_cfg.validate_cfg(cfg=cfg, schema=schema))
        cfg["some_array"].append(5.4)
        self.assertFalse(json_cfg.validate_cfg(cfg=cfg, schema=schema))
    """

    def test_init(self):
        cfg = Config(os.path.join(test_dir, "test_cfg.json"), os.path.join(test_dir, "test_cfg.schema.json"))
        self.assertIsInstance(cfg, Config)
        self.assertTrue(cfg.cfg["name"] == "Test")
        self.assertTrue(cfg.cfg["pi"] == 3.14159)
        self.assertTrue(cfg.cfg["something"]["thing_one"] == "one")
        self.assertTrue(len(cfg.cfg["some_array"]) == 4)

    def test_default_init(self):
        cfg = Config(os.path.join(proj_dir, "config", "default_cfg.json"))
        self.assertIsInstance(cfg, Config)
        self.assertTrue(cfg.cfg["name"] == "default")


if __name__ == "__main__":
    unittest.main()
