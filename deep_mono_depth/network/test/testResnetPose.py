import unittest
import os, sys

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
from ResnetPose import ResNetPoseNet


class TestResPose(unittest.TestCase):
    model = ResNetPoseNet()

    def test_init_posenet(self):
        self.assertIsInstance(self.model, ResNetPoseNet)

    # TODO run a tensor test an then backwards()


if __name__ == "__main__":
    unittest.main()
