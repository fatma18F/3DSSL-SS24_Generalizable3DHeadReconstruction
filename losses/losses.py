import pytorch_lightning as L
from data_processing.dataset import FaceDataset
from model.model_collection import ModelCollection
from gaussian_splatting.scene.mlp_gaussian_model import GaussianModel
from gaussian_splatting.arguments import OptimizationParams, PipelineParams2
import torch
from torch import nn
import time
from tqdm import tqdm
from dreifus.image import Img
import lpips
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.functional.regression import pearson_corrcoef
from dreifus.vector import Vec3

import matplotlib.pyplot as plt
from math import exp
import os

from dreifus.camera import CameraCoordinateConvention

class Losses:
    def __init__(self, config, gauss_predictor):
        super().__init__()
        self.config = config
        self.gauss_predictor = gauss_predictor
        
        self.lpips = lpips.LPIPS(net='alex').cuda() 
        #self.lpips = lpips.LPIPS(net='vgg').cuda() 
        self.lambda_l1 = config['training']['lambda_l1']
        self.lambda_ssim = config['training']['lambda_ssim']
        self.lambda_lpips = config['training']['lambda_lpips']
        self.lambda_depth = config['training']['lambda_depth']
        self.lambda_gaussian_normalization = config['training']['lambda_gaussian_normalization']
        
        self.log_error_images_every_n_steps = config['logging']['log_error_images_every_n_steps']
        self.max_scale = 0
        self.max_displacement = 0
        
    
    def compute_loss(self, data_split, sequence_id, rendered_images, target_images, rendered_depths, target_depths, predicted_gaussians):        
        l1_losses = self.compute_l1_loss(data_split, rendered_images, target_images)
        ssim_losses = self.compute_ssim_loss(data_split, rendered_images, target_images)
        lpips_losses = self.compute_lpips_loss(data_split, rendered_images, target_images)
        limit_gaussians_loss = self.compute_limit_gaussians_loss(data_split, predicted_gaussians)
        
        if self.gauss_predictor.use_depth and rendered_depths is not None and target_depths is not None:
            depth_losses = self.compute_depth_loss(data_split, rendered_images, rendered_depths, target_depths)
        else:
            depth_losses = torch.tensor(0.0).to(self.gauss_predictor.device)
        
        loss = self.lambda_l1 * l1_losses.mean() + self.lambda_ssim * ssim_losses.mean() + self.lambda_lpips * lpips_losses.mean() + self.lambda_depth * depth_losses.mean() + self.lambda_gaussian_normalization * limit_gaussians_loss
        self.gauss_predictor.log(f'{data_split}/loss', loss, batch_size=1)
        
        if (self.log_error_images_every_n_steps > 0 and self.gauss_predictor.global_step % self.log_error_images_every_n_steps == 0) or data_split == 'val':
            self.save_error_images(data_split, sequence_id, rendered_images.keys(), 'loss_l1', l1_losses)
            self.save_error_images(data_split, sequence_id, rendered_images.keys(), 'loss_ssim', ssim_losses)
            #if self.use_depth:
            #    self.save_error_images(data_split, sequence_id, rendered_images.keys(), 'loss_depth', depth_losses)
            
        return loss
    
    def compute_limit_gaussians_loss(self, data_split, predicted_gaussians):
        if self.lambda_gaussian_normalization == 0:
            return 0
        xyz_delta = predicted_gaussians['xyz_delta']
        
        scaling = predicted_gaussians['scaling']
        limit_gaussians_loss = xyz_delta.square().mean() + scaling.square().mean()
        self.gauss_predictor.log(f'{data_split}/loss_limit_gaussians', limit_gaussians_loss.mean(), batch_size=1)

        # Calculate the displacement loss
        displacement_magnitude = torch.norm(xyz_delta, dim=-1)
        displacement_loss = torch.mean(torch.clamp(displacement_magnitude - self.max_displacement, min=0.0) ** 2)
        
        self.gauss_predictor.log(f'{data_split}/displacement_loss', limit_gaussians_loss.mean(), batch_size=1)
        return displacement_loss #limit_gaussians_loss
    
    def compute_l1_loss(self, data_split, rendered_images, target_images):
        l1_losses = []
        for serial, rendered_image in rendered_images.items():
            target_image = target_images[serial]
            l1_loss_for_image = (rendered_image - target_image).abs()
            l1_losses.append(l1_loss_for_image.mean(dim=0).unsqueeze(0))
        
        l1_losses = torch.cat(l1_losses, dim=0).to(self.gauss_predictor.device)
        self.gauss_predictor.log(f'{data_split}/loss_l1', l1_losses.mean(), batch_size=1)
        return l1_losses
    
    def compute_ssim_loss(self, data_split, rendered_images, target_images):
        ssim_losses = []
        
        for serial, rendered_image in rendered_images.items():
            target_image = target_images[serial]
            ssim_loss_for_image = 1.0 - Losses.ssim(rendered_image, target_image)
            ssim_losses.append(ssim_loss_for_image.mean(dim=0).unsqueeze(0))
            
        ssim_losses = torch.cat(ssim_losses, dim=0).to(self.gauss_predictor.device)
        self.gauss_predictor.log(f'{data_split}/loss_ssim', ssim_losses.mean(), batch_size=1)
        return ssim_losses
    
    def compute_lpips_loss(self, data_split, rendered_images, target_images):
        lpips_losses = []
        for serial, rendered_image in rendered_images.items():
            target_image = target_images[serial]
            lpips_loss_for_image = self.lpips(rendered_image.float().cuda(), target_image.float().cuda())
            lpips_losses.append(lpips_loss_for_image.view(1))

        lpips_losses = torch.cat(lpips_losses, dim=0).to(self.gauss_predictor.device)
        self.gauss_predictor.log(f'{data_split}/loss_lpips', lpips_losses.mean(), batch_size=1)
        return lpips_losses
    
    def compute_depth_loss(self, data_split, rendered_images, rendered_depths, target_depths):
        depth_losses = []
        for serial, _ in rendered_images.items():
            gt_depth = torch.from_numpy(target_depths[serial]).float().cuda()
            
            #Monocular depth
            #midas_depth = target_depths[serial].float().cuda()

            rendered_depths = rendered_depths.reshape(-1, 1)
            gt_depth = gt_depth.reshape(-1, 1)


            depth_loss= min(
                            (1 - pearson_corrcoef(-gt_depth, rendered_depths)),
                            (1 - pearson_corrcoef(1 / (gt_depth + 200.), rendered_depths))
            )
            if not torch.isnan(depth_loss):
               depth_losses.append(depth_loss)
            #else:
            #  depth_losses.append(torch.tensor(0.0).to(self.gauss_predictor.device))

        if depth_losses:
           depth_losses = torch.stack(depth_losses).to(self.gauss_predictor.device)
           #depth_losses = depth_losses.mean() 
           self.gauss_predictor.log(f'{data_split}/loss_depth', depth_losses.clone().mean())
        else:
           depth_losses=torch.tensor(0.0).to(self.gauss_predictor.device)
        return depth_losses
    
    def save_error_images(self, data_split, sequence_id, serials, loss_name, losses):
        for i, serial in enumerate(serials):
         if i<3: 
            loss = losses[i].clone().detach().cpu().numpy()
            fig_name = f"{sequence_id.replace('/', '_')}_{serial}_{loss_name}"
            plt.imshow(loss, cmap='hot')
            plt.title(fig_name)
            plt.colorbar()

            path = f"./error_images/{self.gauss_predictor.logger.experiment.id}/{data_split}/{self.gauss_predictor.global_step:06}"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{fig_name}.png")
            plt.close()
    
    @classmethod
    def gaussian(cls, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @classmethod
    def create_window(cls, window_size, channel):
        _1D_window = Losses.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @classmethod
    def ssim(cls, img1, img2, window_size=11, size_average=True):
        channel = img1.size(-3)
        window = Losses.create_window(window_size, channel)
        #print( 'img2 type',img2.dtype)
        #print( 'img1 type',img1.dtype)

        # Ensure both images are of the same type
        img1 = img1.type(torch.float64)
        #img2 = img2.type(torch.float64)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return Losses._ssim(img1, img2, window, window_size, channel, size_average)

    @classmethod
    def _ssim(cls, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


        return ssim_map
        # if size_average:
        #     return ssim_map.mean()
        # else:
        #     return ssim_map.mean(1).mean(1).mean(1)