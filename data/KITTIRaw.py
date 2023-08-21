"""!
@file KITTIRaw.py
@brief Dataset for the KITTI Raw dataset
@author Jason Epp
"""

import os, re, time
import logging
import numpy as np

from typing import Optional, Union

from torch.utils.data import Dataset

# Directory structure of drive in KITTI Raw dataset
oxts_dir = "oxts"
pos_dir = os.path.join(oxts_dir, "data")
pos_time = os.path.join(oxts_dir, "timestamps.txt")
data_dir = "image_02"
img_dir = os.path.join(data_dir, "data")
img_time = os.path.join(data_dir, "timestamps.txt")


class KITTIRaw(Dataset):
    """!
    @brief Pytorch Dataset for the KITTI raw dataset format as downloaded from
           https://www.cvlibs.net/datasets/kitti/raw_data.php
    @param root_dir: directory or list of directories in the folder format as downloaded from above link
    @param K: intrinsic matrix, 4x4 dimensions
    @param Tcam2pos: Transformation from camera to GPS
    @param stride: the number of images between sequence
    @param size: tuple of form (height, width) or None for original size
                 (note image dimension must be divisible by 32)
    """

    def __init__(self, root_dir: str, K: np.array, Tcam2pose: np.array, size: tuple, stride: int = 1):
        super().__init__()
        self.train_list = []
        self.K = K
        self.Tcam = Tcam2pose
        self.stride = stride
        self.height = size[0]
        self.width = size[1]
        if (self.height % 32 != 0) or (self.width % 32 != 0):
            raise ValueError("KITTIRaw: Error, Image dimensions not divisible by 32")
        self.load_training_data(root_dir)

    """!
    @brief Sets the training list based on the root directoy of KITTI Raw Dataset
    @param dir: root directory where drives are
    """

    def load_training_data(self, dir: str):
        seq_list = []
        for drive_date in os.scandir(os.path.abspath(dir)):
            logging.debug(f"Date Directory: {drive_date.path}")
            for drive in os.scandir(drive_date.path):
                # Find all the different drives in the directory
                if drive.is_dir() and "_sync" in drive.name:
                    seq_list.append(drive.path)
                    # sort list in numerical order of sequence
                    # seq_list.sort(key=lambda f: int(re.sub("\D", "", f)))
        logging.debug(f"Found {len(seq_list)} sequences")
        for data_dir in seq_list:
            self.get_training_list(data_dir)

    """!
    @brief extracts dictionary of training references and adds them to a listn
           this includes images sequences and timestams as well as gps position data
    @param seq_dir: the directory if the sequence in the KITTI Raw dataset
    """

    def get_training_list(self, seq_dir):
        img_timestamp = self.parse_timestamps(os.path.join(seq_dir, img_time))
        img_path = os.path.join(seq_dir, img_dir)
        img_list = next(os.walk(img_path), (None, None, []))[2]
        img_list.sort(key=lambda f: int(re.sub("\D", "", f)))
        pos_timestamp = self.parse_timestamps(os.path.join(seq_dir, pos_time))
        pos_path = os.path.join(seq_dir, pos_dir)
        pos_list = next(os.walk(pos_path), (None, None, []))[2]
        pos_list.sort(key=lambda f: int(re.sub("\D", "", f)))
        if len(img_list) - (2 * self.stride) <= 0:
            raise ValueError("KITTIRaw: Error, stride too large for size of datatset")
        for i in range(len(img_list) - (2 * self.stride)):
            train_dict = {
                "image": os.path.join(img_path, img_list[i + self.stride]),
                "img_time": img_timestamp[i + self.stride],
                "sequence": (
                    os.path.join(img_path, img_list[i]),
                    os.path.join(img_path, img_list[i + (2 * self.stride)]),
                ),
                "seq_time": (img_timestamp[i], img_timestamp[i + (2 * self.stride)]),
                "img_pos": os.path.join(pos_path, pos_list[i + self.stride]),
                "pos_time": pos_timestamp[i + self.stride],
                "pos_seq": (
                    os.path.join(pos_path, pos_list[i]),
                    os.path.join(pos_path, pos_list[i + (2 * self.stride)]),
                ),
                "pos_seq_time": (pos_timestamp[i], pos_timestamp[i + (2 * self.stride)]),
            }
            self.train_list.append(train_dict)

    """!
    @brief parse KITTI timestamp file
    @return (list) timestamps
    """

    def parse_timestamps(self, timestamp_file: str) -> list:
        timestamps = []
        logging.debug(f"Timestamp Files: {timestamp_file}")
        times = open(timestamp_file)
        for line in times:
            timestamps.append(self.date_to_epoch(line))
        return timestamps

    """!
    @brief convert data string to epoch
    @return (float) seconds since epoch
    """

    def date_to_epoch(self, date: str) -> float:
        ns = float(date.split(".")[-1])
        timestamp = time.strptime(date[:-4], "%Y-%m-%d %H:%M:%S.%f")
        return time.mktime(timestamp) + ns * 1e-9

    def __len__(self):
        return len(self.train_list)


if __name__ == "__main__":
    logging.basicConfig(
        encoding="utf-8", level=logging.DEBUG, format="[%(levelname)s](%(name)s) %(asctime)s - %(message)s"
    )
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
    test_dir = os.path.join(os.path.abspath("./test"), "test_data", "KITTI_test")
    data = KITTIRaw(root_dir=test_dir, K=K, Tcam2pose=T_ic, size=(320, 1024))
    pass
