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
from deep_mono_depth.network.ResnetUnet import ResnetModel, ResNetUNet
from deep_mono_depth.network.ResnetPose import ResNetPoseNet
# datasets
from deep_mono_depth.data.KITTIRaw import KITTIRaw
from deep_mono_depth.data.GenericData import GenericSupervised
# loss
from deep_mono_depth.core.loss import DepthLoss, ReprojectionLoss, Scaler

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
        #if "pose_network" not in self.cfg["network"].keys() or self.cfg["network"]["pose_network"] == "none":
        #    network["pose"] = None
        #el
        if "pose_network" in self.cfg["network"].keys():
            if self.cfg["network"]["pose_network"] == "pose_resnet18":
                network["pose"] = ResNetPoseNet()
            else:
                raise ("Error Config: Unknown network type, " + self.cfg["network"]["pose_network"])
        if torch.cuda.is_available():
            print(network.keys())
            for key in network.keys():
                network[key] = network[key].to('cuda:0')
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
            num_loaders = self.cfg["dataset"]["data_loaders"] if "data_loaders" in self.cfg["dataset"].keys() else 2
            if self.cfg["dataset"]["data"] == "generic":
                data = GenericSupervised(
                    data_dirs=self.cfg["dataset"]["path"],
                    size=(self.cfg["dataset"]["img_height"],self.cfg["dataset"]["img_width"]),
                    scale_factor=1000,)
                return DataLoader(
                    dataset=data,
                    batch_size=self.cfg["dataset"]["batch_size"],
                    shuffle=True,
                    num_workers=num_loaders,
                    pin_memory=True,
                )
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
                    num_workers=num_loaders,
                    pin_memory=True,
                )
            elif self.cfg["dataset"]["data"] == "kinect":
                raise ("Error Config: Kinect dataset current unsupported")
            else:
                raise ("Error Config: Unknown dataset: " + self.cfg["dataset"]["data"])
        else:
            raise ("Error Config: Dataset configuration not provided in config")

    def get_loss_criteria(self) -> dict[str, torch.nn.Module]:
        loss_functions = {}
        if self.cfg['method'] == "supervised":
            loss_functions['depth'] = DepthLoss().to('cuda:0') if torch.cuda.is_available() else DepthLoss()
        else:
            loss_functions['reprojection'] = ReprojectionLoss()
        return loss_functions

    def get_output_scaler(self):
        if ("min_distance"  in self.cfg.keys()) and ("max_distance" in self.cfg.keys()):
            return Scaler(self.cfg["min_distance"], self.cfg["max_distance"])
        else:
            # no scaling
            return Scaler(0, 1)


if __name__ == "__main__":
    print("JSON Config")
    print(Config.schema_path)
    default_cfg = os.path.abspath(os.path.join( "config", "default_cfg.json"))
    print(default_cfg)
    print("Load Config")
    cfg = Config(default_cfg)
    print("Algoritm type")
    print(cfg.cfg["optimizer"]["algorithm"])
