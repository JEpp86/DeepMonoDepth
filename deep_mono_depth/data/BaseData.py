"""!
@file BaseData.py
@brief Dataset base class for mono depth training
@author Jason Epp
"""

import torch
from torch.utils.data import Dataset

class BaseData(Dataset):
    training_list = []
    def __init__(self) -> None:
        super().__init__()

    @property
    def has_pose(self) -> bool:
        if len(self.training_list) > 0:
            return "transform" in self.training_list[0].keys()
        return False

    @property
    def has_sequence(self) -> bool:
        if len(self.training_list) > 0:
            return "sequence" in self.training_list[0].keys()
        return False

    @property
    def has_gt(self) -> bool:
        return "gt" in self.training_list[0].keys()

    @property
    def has_stereo(self) -> bool:
        return "stereo" in self.training_list[0].keys()
