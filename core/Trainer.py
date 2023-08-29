"""!
@file Trainer.py
@brief Trainign class to train model pased on cfg input
@author Jason Epp
"""
import torch

from util.json_cfg import Config


class Trainer:
    """!
    @brief Training class based on config file
    @param config: Dictionary with training configuration. Loaded from JSON, verified agains schema.
                   See util.json_cfg (util/json_cfg.py) for configuration utilities,
                   See util/schema/cfg.schema.json for schema file
    """

    def __init__(self, config: Config, verbose: bool = True):
        # training params
        self.name = config.cfg["name"]
        self.method = config.cfg["method"]
        self.epochs = config.cfg["epochs"]
        # network
        self.depth_net = config.get_depth_model()
        model_params = list(self.depth_net.parameters())
        self.pose_net = config.get_pose_model()
        if verbose:
            print("Training Params:")
            print("\tName: " + self.name)
            print("\tMethod: " + self.method)
            print("\tEpochs: " + str(self.epochs))
            print("Network:")
            print("\tDepth Model: " + config["network"]["depth_network"])
        if self.method != "supervised":
            if verbose:
                print("\tPose Model: " + config["network"]["pose_network"])
            if self.pose_net == None:
                raise RuntimeError(
                    "Trainer: Error, cannot Train self-supervised without Pose Network, check configuration file"
                )
            else:
                model_params += list(self.pose_net.parameters())
        elif self.pose_net != None:
            if verbose:
                print(
                    "\tWarning: Pose model "
                    + config.cfg["network"]["pose_network"]
                    + " will not be included in supervised training"
                )
        # optimizer
        self.learning_rate = config.cfg["optimizer"]["learning_rate"]
        self.optim = config.get_optimizer(model_params)
        # dataset
        self.batch_size = config.cfg["dataset"]["batch_size"]
        # TODO
        # self.dataset = json_cfg.get_dataloader(conficonfig["dataset"]["data"])
        if verbose:
            print("Optimizer:")
            print("\tAlgorithm: " + config.cfg["optimizer"]["algorithm"])
            print("\tLearning Rate: " + str(self.learning_rate))
            print("Data:")
            print("\tDataset: " + config.cfg["dataset"]["data"])
            print("\tBatch Size: " + str(self.batch_size))
