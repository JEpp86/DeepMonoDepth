"""!
@file KITTIRaw.py
@brief Dataset for the KITTI Raw dataset
@author Jason Epp
"""

import os, re, time
import numpy as np

from typing import Optional, Union

from torch.utils.data import Dataset

# Directory structure of drive in KITTI Raw dataset
pos_dir = 'oxts/data'
pos_time = 'oxts/timestamps.txt'
img_dir = 'image_02/data'
img_time = 'image_02/timestamps.txt'

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
        for drive in os.scan_dir(dir):
             # Find all the different drives in the directory
            if drive.is_dir() and "_sync" in drive:
                # Find all sequences in dataset
                seq_list = []
                for seq in os.scandir(drive):
                    if dir.is_dir():
                        seq_list.append(seq)
                # sort list in numerical order of sequence
                seq_list.sort(key=lambda f: int(re.sub('\D', '', f)))
                for data_dir in seq_list:
                    self.get_training_list(data_dir)

    """!
    @brief extracts dictionary of training references and adds them to a listn
           this includes images sequences and timestams as well as gps position data
    @param seq_dir: the directory if the sequence in the KITTI Raw dataset
    """
    def get_training_list(self, seq_dir):
        img_timestamp = self.parse_timestamps_timestamps(os.path.join(seq_dir, img_time))
        img_path = os.path.join(seq_dir, img_dir)
        img_list = next(os.walk(img_path), (None, None, []))[2]
        img_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        pos_timestamp = self.parse_timestamps(os.path.join(seq_dir, pos_time))
        pos_path = os.path.join(seq_dir, pos_dir)
        pos_list = next(os.walk(pos_path), (None, None, []))[2]
        pos_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        if (len(img_list)-(2*self.stride) <= 0 ):
            raise ValueError("KITTIRaw: Error, stride too large for size of datatset")
        for i in range(len(img_list)-(2*self.stride)):
            train_dict = { 'image': os.path.join(img_path, img_list[i+self.stride]) ,
                            'img_time': img_timestamp[i+self.stride],
                            'sequence': (os.path.join(img_path,img_list[i]),
                                        os.path.join(img_path,img_list[i+(2*self.stride)])),
                            'seq_time': (img_timestamp[i], img_timestamp[i+(2*self.stride)]),
                            'img_pos': os.path.join(pos_path, pos_list[i+self.stride]),
                            'pos_time': pos_timestamp[i+self.stride],
                            'pos_seq': (os.path.join(pos_path,pos_list[i]),
                                        os.path.join(pos_path,pos_list[i+(2*self.stride)])),
                            'pos_seq_time': (pos_timestamp[i], pos_timestamp[i+(2*self.stride)]) }
            self.train_list.append(train_dict)


    """!
    @brief parse KITTI timestamp file
    @return (list) timestamps
    """
    def parse_timestamps(self, timestamp_file: str) -> list:
        timestamps = []
        times = open(timestamp_file)
        for line in times:
            timestamps.append(self.date_to_epoch(line))
        return timestamps

    """!
    @brief convert data string to epoch
    @return (float) seconds since epoch
    """
    def date_to_epoch(self, date: str) -> float:
        ns = float(date.split('.')[-1])
        timestamp = time.strptime(date[:-4], '%Y-%m-%d %H:%M:%S.%f')
        return time.mktime(timestamp) + ns*1e-9

    def __len__(self):
        return len(self.train_list)
