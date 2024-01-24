import json
import jsonschema
import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.abspath(".."))

# networks
from network.ResnetUnet import ResnetModel, ResNetUNet
from network.ResnetPose import ResNetPoseNet

# datasets
from data.KITTIRaw import KITTIRaw


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

    def get_model(self) -> dict[str, torch.nn.Module]:
        network = {}
        if "network" not in self.cfg.keys():
            raise ("Error Config: Network configuration not provided in configuration")
        # Depth network
        if "depth_network" in self.cfg["network"].keys():
            if self.cfg["network"]["depth_network"] == "unet_resnet18":
                network["depth"] = ResNetUNet(ResnetModel.RESNET_18)
            elif self.cfg["network"]["depth_network"] == "unet_resnet34":
                network["depth"] = ResNetUNet(ResnetModel.RESNET_34)
            else:
                raise ("Error Config: Unknown network type, " + self.cfg["network"]["depth_network"])
        else:
            raise ("Error Config: Depth Network not provided in configuration")
        # Pose network
        if "pose_network" not in self.cfg["network"].keys() or self.cfg["network"]["pose_network"] == "none":
            network["pose"] = None
        elif self.cfg["network"]["pose_network"] == "pose_resnet18":
            network["pose"] = ResNetPoseNet()
        else:
            raise ("Error Config: Unknown network type, " + self.cfg["network"]["depth_network"])
        return network

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
                raise ("Error Config: Unknown network type, " + self.cfg["network"]["depth_network"])
        else:
            raise ("Error Config: Network configuration not provided in file")

    def get_optimizer(self, params: list, pose: bool = False) -> Union[torch.optim.SGD, torch.optim.Adam]:
        key = "pose_optimizer" if pose else "optimizer"
        if key in self.cfg.keys():
            if self.cfg[key]["algorithm"] == "sgd":
                return torch.optim.SGD(params=params, lr=self.cfg[key]["learning_rate"])
            elif self.cfg[key]["algorithm"] == "adam":
                return torch.optim.Adam(params=params, lr=self.cfg[key]["learning_rate"])
            else:
                raise ("Error Config: Unknown optimization algoritm in configuration file")
        else:
            raise ("Error Config: Optimizer configuration not provided in file")

    def get_dataloader(self) -> DataLoader:
        if "dataset" in self.cfg.keys():
            if self.cfg["dataset"]["data"] == "generic":
                raise ("Error Config: Generic dataset current unsupported")
            elif self.cfg["dataset"]["data"] == "kitti":
                data = KITTIRaw(
                    root_dir=self.cfg["dataset"]["path"],
                    Tcam2pose=None
                    if "transform" not in self.cfg["dataset"].keys()
                    else self.cfg["dataset"]["transform"],
                    size=(self.cfg["dataset"]["img_height"], self.cfg["dataset"]["img_width"]),
                )
                return DataLoader(
                    dataset=data,
                    batch_size=self.cfg["dataset"]["batch_size"],
                    shuffle=True,
                    num_workers=2,  # (os.cpu_count() - 1),
                    pin_memory=True,
                )
            elif self.cfg["dataset"]["data"] == "kinect":
                raise ("Error Config: Kinect dataset current unsupported")
        else:
            raise ("Error Config: Dataset configuration not provided in config")


if __name__ == "__main__":
    print("JSON Config")
    print(Config.schema_path)
    default_cfg = os.path.abspath(os.path.join("..", "config", "default_cfg.json"))
    print(default_cfg)
    print("Load Config")
    cfg = Config(default_cfg)
    print("Algoritm type")
    print(cfg.cfg["optimizer"]["algorithm"])
