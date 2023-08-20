import unittest
import os, sys
import numpy as np

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)

from KITTIRaw import KITTIRaw

# GPS to velodyne
T_iv = np.array(
    [
        [9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01],
        [-7.854027e-04, 9.998898e-01, -1.482298e-02, 3.195559e-01],
        [2.024406e-03, 1.482454e-02, 9.998881e-01, -7.997231e-01],
        [0, 0, 0, 1],
    ]
)
# Velodyne to Origin
T_vo = np.array(
    [
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
        [0, 0, 0, 1],
    ]
)
# Origin to Camera 2
T_oc = np.array(
    [
        [9.999758e-01, -5.267463e-03, -4.552439e-03, 5.956621e-02],
        [5.251945e-03, 9.999804e-01, -3.413835e-03, 2.900141e-04],
        [4.570332e-03, 3.389843e-03, 9.999838e-01, 2.577209e-03],
        [0, 0, 0, 1],
    ]
)
# GPS to Camera 2
T_ic = np.matmul(np.matmul(T_iv, T_vo), T_oc)
# Intrinsic Matrix
K = np.array(
    [
        [718.856, 0, 607.1928, 0],
        [0, 718.856, 185.2157, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


class TestKITTIRaw(unittest.TestCase):
    def test_init_KITTIraw(self):
        data = KITTIRaw(
            root_dir=os.path.join(test_dir, "test_data"),
            K=K,
            Tcam2pose=T_ic,
            size=(320, 1024),
        )
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
