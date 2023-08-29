import unittest
import os, sys

import torch

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
from ResnetUnet import ResNetUNet, ResnetModel


class TestResDepth(unittest.TestCase):
    model18 = ResNetUNet(ResnetModel.RESNET_18)
    model34 = ResNetUNet(ResnetModel.RESNET_34)

    def test_init_depthnet(self):
        self.assertIsInstance(self.model18, ResNetUNet)
        self.assertIsInstance(self.model34, ResNetUNet)


if __name__ == "__main__":
    unittest.main()
