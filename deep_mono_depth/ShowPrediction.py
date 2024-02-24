import argparse
import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from matplotlib import pyplot as plt
from deep_mono_depth.util.json_cfg import Config

def main():
    parser = argparse.ArgumentParser(description="Train Deep Mono Depth Model")
    parser.add_argument("--config", "-c", type=str, help="configuration file", required=True)
    parser.add_argument("--weights", '-w', type=str, help="Model weights pth file", required=True)
    parser.add_argument("--path", '-p', type=str, help="Image directory/file path", required=True)
    args = parser.parse_args()
    cfg = Config(os.path.abspath(args.config))
    weights = torch.load(args.weights)
    model = cfg.get_model()
    model["depth"].load_state_dict(weights)
    if os.path.isdir(args.path):
        img_files = os.listdir(args.path)
    else:
        img_files = [os.path.abspath(args.path)]
    xfrm = Resize([cfg.cfg['dataset']['img_height'],cfg.cfg['dataset']['img_width']], antialias=True)
    for img_path in img_files:
        img = read_image(img_path) / 255
        img = xfrm(img)
        if torch.cuda.is_available():
            img = img.to("cuda:0")
        with torch.no_grad():
            pred = model["depth"](img.unsqueeze(0))
        _, ax = plt.subplots(1,2)
        ax[0].imshow(img.cpu().permute(1,2,0))
        ax[1].imshow(pred[0].cpu().squeeze().numpy())
        plt.show()






if __name__ == "__main__":
    main()