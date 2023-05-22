import torch

class Trainer():
    def __init__(self, config: dict[str, any]):
        # training params 
        self.name = config['name']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['optimizer']['learning_rate']
        self.method = config['dataset']['method']
        # network
        self.depth_net = self.getModelFromCfg(config)
        if self.method != 'supervised':
            if 'pose_network' not in config['network']:
                raise("Need Pose Network for for self-supervised training methods")
        # optimizer
        if config['optimizer']['algorithm'] == "adam":
            print("Adam Optimizer")
            # self.optim = torch.optim.Adam()

    def getModelFromCfg(self, config: dict[str, any]):
        if config['network']['depth_network'] == "unet_resnet18":
            print("Model Type: " + config['network']['depth_network'])