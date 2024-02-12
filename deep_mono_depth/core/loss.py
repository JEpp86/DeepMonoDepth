"""!
@file loss.py
@brief loss functions
@author Jason Epp
"""

import torch
import numpy as np
from deep_mono_depth.util.geometry import scale_depth, scale_depth_from_disparity

class Scaler(torch.nn.Module):
    def __init__(self, min_distance, max_distance) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance

    def forward(self,input):
        return scale_depth(input, self.min_distance, self.max_distance)


class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target) -> torch.TensorType:
        return torch.abs(target - pred)


class SSIMLoss(torch.nn.Module):
    def __init__(self, kernel_size=3, C1=1e-2**2, C2=3e-2**2):
        super(SSIMLoss, self).__init__()
        self.refl = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.mu_pool_x = torch.nn.AvgPool2d(kernel_size, 1)
        self.mu_pool_y = torch.nn.AvgPool2d(kernel_size, 1)
        self.sig_pool_x = torch.nn.AvgPool2d(kernel_size, 1)
        self.sig_pool_y = torch.nn.AvgPool2d(kernel_size, 1)
        self.sig_pool_xy = torch.nn.AvgPool2d(kernel_size, 1)

        self.C1 = C1
        self.C2 = C2

    def forward(self, pred, target):
        pred = self.refl(pred)
        target = self.refl(target)
        u_pred = self.mu_pool_x(pred)
        u_target = self.mu_pool_y(target)
        # std dev
        sig_pred = self.sig_pool_x(pred**2) - u_pred**2
        sig_targ = self.sig_pool_y(target**2) - u_target**2
        sig_co = self.sig_pool_xy(pred * target) - u_pred * u_target

        ssim_num = (2 * u_pred * u_target + self.C1) * (2 * sig_co + self.C2)
        ssim_den = (u_pred**2 + u_target**2 + self.C1) * (sig_pred + sig_targ + self.C2)
        return torch.clamp(((1 - ssim_num / ssim_den)) / 2, 0, 1)


class EdgeLoss(torch.nn.Module):
    """Used for calcualting edge aware loss between target image and predicted depth map"""

    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred, target):
        # Disparity Gradients
        grad_disp_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_disp_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        # Image Gradients
        grad_img_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class GradientLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(GradientLoss, self).__init__()

    def grad(self, x):
        """Calculate gradient of an image"""
        diff_x = x[..., 1:, :-1] - x[..., :-1, :-1]
        diff_y = x[..., :-1, 1:] - x[..., :-1, :-1]
        mag = torch.sqrt(diff_x**2 + diff_y**2)
        angle = torch.atan(diff_y / (diff_x + 1e-6))

        return mag, angle

    def grad_mask(mask):
        """Calculate gradient mask from image mask"""
        return mask[..., :-1, :-1] & mask[..., 1:, :-1] & mask[..., :-1, 1:] & mask[..., 1:, 1:]

    def forward(self, pred, gt, mask):
        """Forward pass for the gradient loss."""
        pred_grad, gt_grad = self.grad(pred), self.grad(gt)
        mask_grad = self.grad_mask(mask)

        loss_mag = (pred_grad[0][mask_grad] - gt_grad[0][mask_grad]).abs().sqrt()
        loss_ang = (pred_grad[1][mask_grad] - gt_grad[1][mask_grad]).abs()
        # should handle zero crossings
        if loss_ang > torch.pi:
            loss_ang = (2 * torch.pi) - loss_ang

        loss_mag = torch.clip(loss_mag, min=0.0, max=10.0)

        ratio = 0.10
        valid = int(ratio * loss_mag.shape[0])
        loss_mag = loss_mag.sort()[0][valid:-valid].mean()
        loss_ang = loss_ang.sort()[0][valid:-valid:].mean()

        return loss_mag + loss_ang


class PhotometricReprojection(torch.nn.Module):
    def __init__(self, K, T):
        self.K = K
        self.T = T

    def forward(self):
        pass


class DepthLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.85,
        edge_weight: float = 0.1,
        ssim_kernel: int = 3,
        ssim_c1: float = 1e-2**2,
        ssim_c2: float = 3e-2**2,
    ):
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.edge_weight = edge_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss(kernel_size=ssim_kernel, C1=ssim_c1, C2=ssim_c2)
        self.edge_loss = EdgeLoss()

    def forward(self, pred, target) -> float:
        return (
            (self.alpha * self.ssim_loss(pred, target).mean())
            + ((1 - self.alpha) * self.l1_loss(pred, target).mean()) # .mean(1, True))
            + (self.edge_weight * self.edge_loss(pred, target)) # .mean(1, True))
        )


class ReprojectionLoss(torch.nn.Module):
    """Calculates Reprojection Loss"""

    def __init__(self, batch_size: int, height: int, width: int, K: torch.TensorType):
        """
        Parameters:
            batch_size: The size of a data batch
            height: image height
            width: image width
            K: 3x4 Intrinsic matrix
        """
        super(ReprojectionLoss, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.K = K.repeat(self.batch_size, 1, 1)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.sensor_grid = self.generate_sensor_grid()
        self.ssim_loss = SSIMLoss()
        self.L1_loss = L1Loss()
        self.edge_loss = EdgeLoss()

    def generate_sensor_grid(self) -> torch.TensorType:
        """Generates grid of homogeneous camera points
        Returns:
            Tensor of shape [Batch Size, 3 (x, y, z), Pixels(height * width)]
        """
        pix_coords = np.stack(
            np.meshgrid(range(self.width), range(self.height), indexing="xy"), axis=0, dtype=np.float32
        )
        pix_coords = torch.from_numpy(pix_coords)
        pix_coords = torch.unsqueeze(torch.stack([pix_coords[0].view(-1), pix_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(self.batch_size, 1, 1)
        return torch.cat([self.pix_coords, self.ones], 1)

    def project_depth(self, depth: torch.TensorType):
        """Projects Depth from Sensor Grid to Homogeneous camera coordinates
                 [[Xi],     [[Xc],
        K^(-1) *  [Yi],   =  [Yc],
                  [Zc]]      [Zc],
                             [1.]]
        """
        # TODO should depth be muliplied before Grid muliply?
        cam_points = torch.linalg.sove(self.K[:, :3, :3], self.sensor_grid)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        return torch.cat([cam_points, self.ones], 1)

    def transform_points_to_frame(self, T: torch.TensorType) -> torch.TensorType:
        pass

    def forward(
        self, depth: torch.TensorType, sequence: torch.TensorType, transform: torch.TensorType, target: torch.TensorType
    ) -> float:
        return 0.0
