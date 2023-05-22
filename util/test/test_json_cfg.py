import unittest
import os, sys

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
import json_cfg

class TestCfg(unittest.TestCase):
    def test_read(self):
        cfg = json_cfg.load_cfg(os.path.join(test_dir, "test_cfg.json"))
        self.assertTrue(cfg['name']=="Test")
        self.assertTrue(cfg['pi']==3.14159)
        self.assertTrue(cfg['something']['thing_one']=="one")
        self.assertTrue(len(cfg['some_array'])==4)
        
    def test_validate(self):
        schema = json_cfg.load_cfg(os.path.join(test_dir, "test_cfg.schema.json"))
        cfg = json_cfg.load_cfg(os.path.join(test_dir, "test_cfg.json"))
        self.assertTrue(json_cfg.validate_cfg(cfg=cfg, schema=schema))
        cfg["some_array"].append(5.4)
        self.assertFalse(json_cfg.validate_cfg(cfg=cfg, schema=schema))


if __name__ == '__main':
    unittest.main()
