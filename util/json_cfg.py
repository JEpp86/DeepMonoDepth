import json
import jsonschema
import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.abspath(".."))

from network.ResnetUnet import ResnetModel, ResNetUNet
from network.ResnetPose import ResNetPoseNet


class Config:
    schema_path = os.path.join(os.path.dirname(__file__), "schema", "cfg.schema.json")

    def __init__(self, cfg_path: os.path, schema_path: os.path = schema_path) -> None:
        self.cfg = self.load_cfg(cfg_path)
        self.schema = self.load_cfg(schema_path)
        self.validate_cfg

    def load_cfg(self, cfg_path: str) -> json:
        f_cfg = open(os.path.abspath(cfg_path), "r")
        cfg = json.load(f_cfg)
        f_cfg.close()
        return cfg

    def validate_cfg(self):
        try:
            jsonschema.validate(instance=self.cfg, schema=self.schema)
        except:
            raise ("Error Config: Schema validation failed")

    def get_depth_model(self) -> torch.nn.Module:
        if "network" in self.cfg.keys() and "depth_network" in self.cfg["network"].keys():
            if self.cfg["network"]["depth_network"] == "unet_resnet18":
                return ResNetUNet(ResnetModel.RESNET_18)
            elif self.cfg["network"]["depth_network"] == "unet_resnet34":
                return ResNetUNet(ResnetModel.RESNET_34)
            else:
                raise ("Error Config: Unknown network type, " + self.cfg["network"]["depth_network"])
        else:
            raise ("Error Config: Network configuration not provided in file")

    def get_pose_model(self) -> Optional[torch.nn.Module]:
        if "network" in self.cfg.keys():
            if "pose_network" not in self.cfg["network"].keys() or self.cfg["network"]["pose_network"] == "none":
                return None
            elif self.cfg["network"]["pose_network"] == "pose_resnet18":
                return ResNetPoseNet()
            else:
                raise ("Error: Unknown network type, " + self.cfg["network"]["depth_network"])
        else:
            raise ("Error: Network configuration not provided in file")

    def get_optimizer(self, params: list, pose: bool = False) -> Union[torch.optim.SGD, torch.optim.Adam]:
        key = "pose_optimizer" if pose else "optimizer"
        if key in self.cfg.keys():
            if self.cfg[key]["algorithm"] == "sgd":
                return torch.optim.SGD(params=params, lr=self.cfg[key]["learning_rate"])
            elif self.cfg[key]["algorithm"] == "adam":
                return torch.optim.Adam(params=params, lr=self.cfg[key]["learning_rate"])
            else:
                raise ("Error: Unknown optimization algoritm in configuration file")
        else:
            raise ("Error: Optimizer configuration not provided in file")

    def get_dataloader(self) -> DataLoader:
        pass


if __name__ == "__main__":
    print("JSON Config")
    print(Config.schema_path)
    default_cfg = os.path.abspath(os.path.join("..", "config", "default_cfg.json"))
    print(default_cfg)
    print("Load Config")
    cfg = Config(default_cfg)
    print("Algoritm type")
    print(cfg.cfg["optimizer"]["algorithm"])
