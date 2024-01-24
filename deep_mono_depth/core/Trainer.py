"""!
@file Trainer.py
@brief Trainign class to train model pased on cfg input
@author Jason Epp
"""
import torch
import tqdm
import logging
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
        self.network = config.get_model()
        # self.depth_net = config.get_depth_model()
        # if torch.cuda.is_available():
        #    self.depth_net.to("cuda:0")
        model_params = list(self.network["depth"].parameters())
        # self.pose_net = config.get_pose_model()
        # if self.pose_net and torch.cuda.is_available():
        #    self.pose_net.to("cuda:0")
        if verbose:
            print("Training Params:")
            print("\tName: " + self.name)
            print("\tMethod: " + self.method)
            print("\tEpochs: " + str(self.epochs))
            print("Network:")
            print("\tDepth Model: " + config.cfg["network"]["depth_network"])
        if self.method != "supervised":
            if verbose:
                print("\tPose Model: " + config.cfg["network"]["pose_network"])
            if "pose" not in self.network.keys() or self.network["pose"] == None:
                raise RuntimeError(
                    "Trainer: Error, cannot Train self-supervised without Pose Network, check configuration file"
                )
            else:
                model_params += list(self.network["pose"].parameters())
        elif "pose" in self.network.keys() and self.network["pose"] != None:
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
        self.dataset = config.get_dataloader()
        if verbose:
            print("Optimizer:")
            print("\tAlgorithm: " + config.cfg["optimizer"]["algorithm"])
            print("\tLearning Rate: " + str(self.learning_rate))
            print("Data:")
            print("\tDataset: " + config.cfg["dataset"]["data"])
            print("\tBatch Size: " + str(self.batch_size))
        self.tepoch = tqdm.tqdm(range(self.epochs), unit="epoch", position=0, leave=True)
        self.tbatch = tqdm.tqdm(self.dataset, unit="batch", position=1, leave=False)

    def train(self):
        for self.epoch in range(self.epochs):
            # if self.epoch > 0:
            #    self.tbatch.refresh()
            self.run_batch()
            self.tepoch.update()
        # self.tbatch.close()
        # self.tepoch.close()
        logging.info("Training Complete")

    def run_batch(self):
        self.tbatch.reset()
        for batch_idx, inputs in enumerate(self.dataset):
            # calculate loss
            # run optimizer and scheduler steps
            self.tbatch.update()
        self.tbatch.refresh()
