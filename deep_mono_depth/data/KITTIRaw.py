"""!
@file KITTIRaw.py
@brief Dataset for the KITTI Raw dataset
@author Jason Epp
"""

import os, re, time
import logging
import numpy as np
import cv2

from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(".."))

import util.geometry as geo


class KITTIRaw(Dataset):
    """Pytorch Dataset for the KITTI raw dataset format as downloaded from
    https://www.cvlibs.net/datasets/kitti/raw_data.php
    """

    # Directory structure of drive in KITTI Raw dataset
    oxts_dir = "oxts"
    pos_dir = os.path.join(oxts_dir, "data")
    pos_time = os.path.join(oxts_dir, "timestamps.txt")
    data_dir = "image_02"
    img_dir = os.path.join(data_dir, "data")
    img_time = os.path.join(data_dir, "timestamps.txt")

    def __init__(self, root_dir: str, size: tuple[int, int], Tcam2pose: Optional[np.array] = None, stride: int = 1):
        """
        Parameters:
            root_dir: directory or list of directories in the folder format as downloaded from above link
            size: tuple of form (height, width) or None for original size (note image dimension must be divisible by 32)
            Tcam2pos: Transformation from camera to GPS [Optional]
            stride: the number of images between sequence [default = 1]
        """
        super().__init__()
        self.train_list = []
        # self.K = np.array(K, dtype=np.float32).reshape(4, 3)
        self.Tcam = Tcam2pose if (Tcam2pose is None) else np.array(Tcam2pose, dtype=np.float32).reshape(4, 4)
        self.stride = int(stride)
        self.height = int(size[0])
        self.width = int(size[1])
        if (self.height % 32 != 0) or (self.width % 32 != 0):
            raise ValueError("KITTIRaw: Error, Image dimensions not divisible by 32")
        self.resize = transforms.Resize([self.height, self.width], antialias=True)
        try:
            self.load_training_data(root_dir)
        except:
            raise RuntimeError("Invalid KITTI Raw root directory")
        logging.debug(f"Number of training sequences: {len(self.train_list)}")

    def load_training_data(self, dir: str):
        """Sets the training list based on the root directoy of KITTI Raw Dataset
        Paramaters:
            dir: root directory where drives are
        """
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
        for data in seq_list:
            self.get_training_list(data)

    def get_training_list(self, seq_dir):
        """extracts dictionary of training references and adds them to a list, this includes images sequences and timestamps as well as gps position data
        Parameters:
            seq_dir: the directory if the sequence in the KITTI Raw dataset
        """
        img_timestamp = self.parse_timestamps(os.path.join(seq_dir, self.img_time))
        img_path = os.path.join(seq_dir, self.img_dir)
        logging.debug(f"Image path: {img_path}")
        img_list = next(os.walk(img_path), (None, None, []))[2]
        img_list.sort(key=lambda f: int(re.sub("\D", "", f)))
        pos_timestamp = self.parse_timestamps(os.path.join(seq_dir, self.pos_time))
        pos_path = os.path.join(seq_dir, self.pos_dir)
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
        times.close()
        return timestamps

    """!
    @brief convert data string to epoch
    @return (float) seconds since epoch
    """

    def date_to_epoch(self, date: str) -> float:
        ns = float(date.split(".")[-1])
        timestamp = time.strptime(date[:-4], "%Y-%m-%d %H:%M:%S.%f")
        return time.mktime(timestamp) + ns * 1e-9

    # Position in world fra
    def parse_position(self, index, sequence=None):
        if sequence == None:
            f_pos = open(self.train_list[index]["img_pos"])
        else:
            f_pos = open(self.train_list[index]["pos_seq"][sequence])
        pos_str = f_pos.readline()
        f_pos.close()
        pos_lst = pos_str.split(" ")
        x, y = geo.lat_lon2xy(pos_lst[0:2])
        pos = np.array([x, y, pos_lst[2], pos_lst[3], pos_lst[4], pos_lst[5]], dtype=np.float32)
        R1 = geo.generate_rotation_matrix(pos[3:])
        t1 = pos[:3].reshape((3, 1))
        Pg1 = geo.create_transformation_matrix(R1, t1)
        return np.matmul(Pg1, geo.inverse_transform(self.Tcam))

    def get_image(self, index, sequence=None):
        if sequence == None:
            image_path = self.train_list[index]["image"]
        else:
            image_path = self.train_list[index]["sequence"][sequence]
        # TODO replace with PIL, this was handy when working with 16bit depth maps
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(np.array(img).astype(np.float32))
        # if image happens to be monochrome (used previously when converting depth maps to tensors)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)
        # creates floating point image of range [0., 1.]
        img = img / 255.0
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index) -> dict[str, any]:
        if self.Tcam is None:
            p_xfrm = None
            n_xfrm = None
        else:
            cam_pose = self.parse_position(index=index, sequence=None)
            cam_pose_p = self.parse_position(index=index, sequence=0)
            cam_pose_n = self.parse_position(index=index, sequence=1)
            p_xfrm = geo.get_relative_pose(cam_pose_p, cam_pose)
            n_xfrm = geo.get_relative_pose(cam_pose_n, cam_pose)
        img = self.get_image(index=index)
        p_img = self.get_image(index=index, sequence=0)
        n_img = self.get_image(index=index, sequence=1)
        sample = {
            "image": img,
            "image_time": self.train_list[index]["img_time"],
            "sequence": (p_img, n_img),
            "sequence_time": (self.train_list[index]["seq_time"][0], self.train_list[index]["seq_time"][1]),
            "transform": (p_xfrm, n_xfrm),
        }
        return sample


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
    # print(T_ic)
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
    data = KITTIRaw(root_dir=test_dir, size=(320, 1024), Tcam2pose=T_ic)
    # print(len(data))
    pass
