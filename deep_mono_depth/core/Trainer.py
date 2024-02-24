"""!
@file Trainer.py
@brief Trainign class to train model pased on cfg input
@author Jason Epp
"""
import os
import tqdm
import logging
import numpy as np
import torch

from deep_mono_depth.util.json_cfg import Config



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
        model_params = list(self.network["depth"].parameters())
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
                # if using single optimizer TODO multi optimizer case
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
        self.optimizer = config.get_optimizer(model_params)
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
            print("\tNumber of Batches: " + str(len(self.dataset)))
        # loss criteria
        self.loss = config.get_loss_criteria()
        self.avg_loss = 0.
        if len(self.loss) == 0:
            raise("no loss criteria to train ")
        elif verbose:
            print("Loss:")
            for key in self.loss.keys():
                print("\t" + key)
        self.scaler = config.get_output_scaler()
        self.tepoch = tqdm.tqdm(range(self.epochs), unit="epoch", position=0, leave=True)
        self.tbatch = tqdm.tqdm(self.dataset, unit="batch", position=1, leave=False)

    def train(self):
        for self.epoch in range(self.epochs):

            self.run_batch()
            self.save_model()
            self.tepoch.set_postfix(avg_loss=self.avg_loss)
            self.tepoch.update()

        logging.info("Training Complete")

    def run_batch(self):
        self.tbatch.reset()
        self.optimizer.zero_grad()
        for self.batch_idx, inputs in enumerate(self.dataset):
            outputs = self.get_outputs(inputs)
            self.calculate_loss(inputs, outputs)
            self.update_optimizer()
            self.tbatch.update()
        self.tbatch.refresh()

    def calculate_loss(self, input : dict[str, torch.TensorType] , output : dict[str, torch.TensorType] ):
        loss = 0.
        if "depth" in self.loss.keys():
            loss += self.loss['depth'](output['depth'], input['gt'].to('cuda:0'))
        loss.backward()
        self.tbatch.set_postfix(loss=loss.item())
        # rolling average
        self.avg_loss = (self.avg_loss * self.batch_idx + loss.item()) / (self.batch_idx + 1)

    def update_optimizer(self):
        # TODO update optimizer to dictionary, for muli-optimizer configurations
        self.optimizer.step()
        self.optimizer.zero_grad()


    def get_outputs(self, input : dict[str, torch.TensorType]):
        output = {}
        if "depth" in self.network.keys():
            op = self.network["depth"](input["image"].to("cuda:0"))
            # TODO multiscale not handled
            output['depth'] = self.scaler(op[0])
        return output

    def save_model(self):
        model_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for key in self.network.keys():
            file_name = self.name + "_" + key + "_epoch" + str(self.epoch + 1) + ".pth"
            torch.save(self.network[key].state_dict(), os.path.join(model_dir, file_name))


