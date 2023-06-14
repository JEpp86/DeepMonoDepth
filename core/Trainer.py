"""!
@file Trainer.py
@brief Trainign class to train model pased on cfg input
@author Jason Epp
"""
import torch

from util import json_cfg


class Trainer():
    """!
    @brief Training class based on config file
    @param config: Dictionary with training configuration. Loaded from JSON, verified agains schema.
                   See util.json_cfg (util/json_cfg.py) for configuration utilities,
                   See util/schema/cfg.schema.json for schema file
    """
    def __init__(self, config: dict[str, any]):
        # training params
        self.name = config['name']
        self.method = config['method']
        self.epochs = config['epochs']
        print("Training Params:")
        print("\tName: " + self.name)
        print("\tMethod: " + self.method)
        print("\tEpochs: " + str(self.epochs))
        # network
        print("Network:")
        self.depth_net = json_cfg.get_depth_model(config)
        print("\tDepth Model: " + config['network']['depth_network'])
        model_params = list(self.depth_net.parameters())
        self.pose_net = json_cfg.get_pose_model(config)
        if self.method != 'supervised':
            print("\tPose Model: " + config['network']['pose_network'])
            if self.pose_net == None:
                raise RuntimeError("Trainer: Error, \
                                   cannot Train self-supervised without Pose Network, check configuration file")
            else:
                model_params += list(self.pose_net.parameters())
        elif self.pose_net != None:
            print("\tWarning: Pose model " + config['network']['pose_network'] + \
                    " will not be included in supervised training")
        # optimizer
        print("Optimizer:")
        self.learning_rate = config['optimizer']['learning_rate']
        print("\tAlgorithm: " + config['optimizer']['algorithm'])
        print("\tLearning Rate: " + str(self.learning_rate))
        self.optim = json_cfg.get_optimizer(config, model_params)
        # dataset
        print("Data:")
        print("\tDataset: " + config['dataset']['data'])
        self.batch_size = config['dataset']['batch_size']
        print("\tBatch Size: " + str(self.batch_size))

