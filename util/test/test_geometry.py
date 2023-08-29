import unittest
import os, sys
import torch

test_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(test_dir)
sys.path.insert(0, parent_dir)
import geometry


class TestGeometry(unittest.TestCase):
    def test_latlon2xy(self):
        lat = 49.277445
        lon = -122.868058
        utm_zone = geometry.get_UTM_zone(lon)
        self.assertEqual(utm_zone, 10)
        xy = geometry.lat_lon2xy((lat, lon))
        self.assertAlmostEqual(xy[0], 509596.85, delta=0.01)
        self.assertAlmostEqual(xy[1], 5458307.15, delta=0.01)


if __name__ == "__main__":
    unittest.main()
