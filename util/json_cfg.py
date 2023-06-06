import json
import jsonschema
import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from network.ResnetUnet import ResnetModel, ResNetUNet
from network.ResnetPose import ResNetPoseNet

schema_path = os.path.join(os.path.dirname(__file__), "schema", "cfg.schema.json")

def load_cfg(cfg_path: str) -> json:
    f_cfg = open(os.path.abspath(cfg_path), "r")
    cfg = json.load(f_cfg)
    f_cfg.close()
    return cfg

def validate_cfg(cfg: json, schema: json) -> bool:
    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except:
        return False
    return True

def get_depth_model(cfg: json) -> torch.nn.Module:
    if "network" in cfg.keys() and "depth_network" in cfg['network'].keys():
        if cfg['network']['depth_network'] == "unet_resnet18":
            return ResNetUNet(ResnetModel.RESNET_18)
        elif cfg['network']['depth_network'] == "unet_resnet34":
            return ResNetUNet(ResnetModel.RESNET_34)
        else:
            raise("Error: Unknown network type, " + cfg['network']['depth_network'])
    else:
        raise("Error: Network configuration not provided in file")

def get_pose_model(cfg: json) -> Optional[torch.nn.Module]:
    if "network" in cfg.keys():
        if "pose_network" not in cfg['network'].keys() or cfg['network']['pose_network'] == 'none':
            return None
        elif cfg['network']['pose_network'] == "pose_resnet18":
            return ResNetPoseNet()
        else:
            raise("Error: Unknown network type, " + cfg['network']['depth_network'])
    else:
        raise("Error: Network configuration not provided in file")

def get_optimizer(cfg: json, params: list, pose: bool = False) -> Union[torch.optim.SGD, torch.optim.Adam]:
    key = "pose_optimizer" if pose else "optimizer"
    if key in cfg.keys():
        if cfg[key]["algorithm"] == "sgd":
            return torch.optim.SGD(params=params, lr=cfg[key]["learning_rate"])
        elif cfg[key]["algorithm"] == "adam":
            return torch.optim.Adam(params=params, lr=cfg[key]["learning_rate"])
        else:
            raise("Error: Unknown optimization algoritm in configuration file")
    else:
        raise("Error: Optimizer configuration not provided in file")

def get_dataloader(cfg: json) -> DataLoader:
    pass

if __name__ == '__main__':
    print("JSON Config")
    print(schema_path)
    default_cfg = os.path.abspath(os.path.join("..", "config", "default_cfg.json"))
    print(default_cfg)
    print("Load Config")
    cfg = load_cfg(default_cfg)
    print("Algoritm type")
    print(cfg['optimizer']['algorithm'])

