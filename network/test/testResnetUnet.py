import unittest
import os, sys

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
from ResnetUnet import ResNetUNet


class TestResDepth(unittest.TestCase):
    def test_init_depthnet(self):
        self.assertTrue(False)


if __name__ == "__main":
    unittest.main()