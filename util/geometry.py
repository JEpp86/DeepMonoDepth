"""!
@file geometry.py
@brief various geometric functions
@author Jason Epp
"""
import math
import pyproj
import numpy as np
from scipy.spatial.transform import Rotation

"""!@ brief poses and transformations
"""


def generate_rotation_matrix(axis):
    R = Rotation.from_euler("xyz", axis)
    return R.as_matrix()


def create_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def inverse_transform(Transform):
    R_inv = np.eye(4)
    R_inv[:3, :3] = Transform[:3, :3].T
    R_inv[:3, 3:] = np.matmul(-R_inv[:3, :3], Transform[:3, 3:])
    return R_inv


# gets pose of T1 relative to T2
def get_relative_pose(T1, T2):
    P1_2 = np.eye(4)
    R2inv = np.linalg.inv(T2[:3, :3])
    P1_2[:3, :3] = np.matmul(R2inv, T1[:3, :3])
    diff = np.subtract(T2[:3, 3:], T1[:3, 3:])
    P1_2[:3, 3:] = np.matmul(R2inv, diff)
    return P1_2


"""! @brief Geographic Conversion
"""


def lat_lon2xy(pos):
    lon = float(pos[1])
    zone = get_UTM_zone(lon)
    P = pyproj.Proj(proj="utm", zone=zone, ellps="WGS84", preserve_units=True)
    return P(pos[1], pos[0])


def get_UTM_zone(longitude: float) -> int:
    return int(math.ceil((longitude + 180.0) / 6.0))
