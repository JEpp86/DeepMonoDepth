import os, re
from typing import Union

# from deep_mono_depth.data.BaseData import BaseData

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np


class GenericSupervised(Dataset):  # (BaseData):
    def __init__(
        self,
        data_dirs: Union[str, list[str]],
        size: tuple[int, int],
        scale_factor: float,
        image_dirname: str = "img",
        depth_dirname: str = "depth",
    ) -> None:
        super().__init__()
        self.training_list = []
        self.get_training_list(data_dirs, image_dirname, depth_dirname)
        self.height = size[0]
        self.width = size[1]
        self.scale_factor = scale_factor
        if (self.height % 32 != 0) or (self.width % 32 != 0):
            raise ValueError("Dataset: Error, Image dimensions not divisible by 32")
        self.resize = transforms.Resize([self.height, self.width], antialias=True)

    def get_training_list(self, data_path: list[str], img_directory: str, depth_directory: str) -> None:
        if isinstance(data_path, str):
            root_path = data_path
            data_path = []
            if img_directory not in os.listdir(root_path):
                for file in os.listdir(root_path):
                    if os.path.isdir(os.path.join(root_path, file)):
                        data_path += [os.path.join(root_path, file)]
            else:
                data_path = [root_path]
            # python absurdities
            while img_directory not in [
                d for d in os.listdir(data_path[0]) if os.path.isdir(os.path.join(data_path[0], d))
            ]:
                tmp_path = data_path
                data_path = []
                for d_dir in tmp_path:
                    data_path += [
                        os.path.join(d_dir, d) for d in os.listdir(d_dir) if os.path.isdir(os.path.join(d_dir, d))
                    ]

        for data_dir in data_path:
            img_dir = os.path.join(data_dir, img_directory)
            img_list = next(os.walk(img_dir), (None, None, []))[2]
            img_list.sort(key=lambda f: int(re.sub("\D", "", f)))
            depth_dir = os.path.join(data_dir, depth_directory)
            depth_list = next(os.walk(depth_dir), (None, None, []))[2]
            depth_list.sort(key=lambda f: int(re.sub("\D", "", f)))
            if len(img_list) != len(depth_list):
                raise IndexError("Dataset: Error, mismatched number of images and ground truth at " + data_dir)
            for i in range(len(img_list)):
                train_dict = {
                    "image": os.path.join(img_dir, img_list[i]),
                    "depth": os.path.join(depth_dir, depth_list[i]),
                }
                self.training_list.append(train_dict)
                # print("Added image to list")
        if len(self.training_list) == 0:
            raise ValueError("Dataset: No Image data provided")

    def image_to_tensor(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            # depth image, single channel
            img = torch.tensor(np.array(img).astype(np.float32))
            img = img.unsqueeze(0)
        else:
            # convert image from cv2 format to tensor format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(np.array(img).astype(np.float32))
            img = img.permute(2, 0, 1)
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.training_list)

    def __getitem__(self, index) -> dict[str, any]:
        # print("---")
        # print(index)
        # print(self.training_list)
        depth = self.image_to_tensor(self.training_list[index]["depth"])
        depth = depth / self.scale_factor
        img = self.image_to_tensor(self.training_list[index]["image"])
        img = img / 255.0
        sample = {"image": img, "gt": depth}
        return sample


class GenericSelfSupervised(Dataset):  # (BaseData):
    def __init__(self) -> None:
        super().__init__()
        pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    test_dir1 = os.path.abspath(
        "E:\\Kinect"
    )  # os.path.join(os.path.abspath("./data/test"), "test_data", "Kinect_test") # , "td1")
    # test_dir2 = os.path.join(os.path.abspath("./test"), "test_data", "Kinect_test", "td2")
    data = GenericSupervised(test_dir1, size=(768, 1024), scale_factor=1000)
    # plt.imshow(data[0]['image'].permute(1,2,0))
    # plt.show()
    # plt.imshow(data[0]['gt'].squeeze(0))
    # plt.show()

    print(len(data))
    data_loader = DataLoader(data, 8, shuffle=True, pin_memory=True)

    for batch_idx, sample in enumerate(data_loader):
        print(batch_idx)
        print(sample["image"].size())
        # print(sample['image'])
        print(sample["gt"].size())
        # print(sample['gt'])
        plt.imshow(sample["image"][0, ...].squeeze(0).permute(1, 2, 0))
        plt.show()
        plt.imshow(sample["gt"][0, ...].squeeze(0).squeeze(0))
        plt.show()
        if batch_idx > 3:
            break
