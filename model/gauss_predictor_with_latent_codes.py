import pytorch_lightning as L
from data_processing.dataset import FaceDataset
from model.model_collection import ModelCollection
from gaussian_splatting.scene.mlp_gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.arguments import OptimizationParams, PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.viewer import GaussianViewer
from model.gauss_predictor import GaussPredictor
import torch
from torch import nn
import time
from tqdm import tqdm

class GaussPredictorWithLatentCodes(GaussPredictor):
    def __init__(self, config, data_module, **kwargs):
        super().__init__(config=config)
    
        # Latent Codes
        config_latent_codes = config['model']['model_parameters']['latent_codes']
        self.data_module = data_module
        self.latent_code_learning_rate = config_latent_codes['latent_code_learning_rate']
        self.test_time_optimization_steps = config_latent_codes['test_time_optimization_steps']
        self.latent_code_size = config_latent_codes['latent_code_size']
        self.training_latent_codes = nn.ParameterDict()
        
        for sequence in self.data_module.train_dataloader().dataset.sequences:
            sequence_id = sequence['id']
            self.training_latent_codes[sequence_id] = nn.Parameter(torch.randn((1, self.latent_code_size)), requires_grad=True)    
    
    def training_step(self, batch, batch_idx):
        sequence_id = batch['sequence_id']
        
        number_of_gaussians = batch['initial_gaussians']['xyz'].shape[0]
        latent_codes = self.training_latent_codes[sequence_id].repeat(number_of_gaussians, 1)
        batch['initial_gaussians']['latent_codes'] = latent_codes
        
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        initial_gaussians = batch['initial_gaussians']
        images = batch['images']
        sequence_id = batch['sequence_id']
        target_depths = batch['depths']
        
        camera_params = batch['camera_params']
        camera_params['cam_2_world_poses'] = FaceDataset.dicts_to_poses(camera_params['cam_2_world_poses'])
        
        input_serials = batch['input_serials']
        input_images = batch['input_images']
        input_cam_2_world_poses = {serial: cam_2_world_pose for serial, cam_2_world_pose in camera_params['cam_2_world_poses'].items() if serial in input_serials}
        input_camera_params = {'cam_2_world_poses': input_cam_2_world_poses, 'intrinsics': camera_params['intrinsics']}
        
        gaussian_model = self.get_gaussian_model(sequence_id)
        self.visualize(sequence_id, camera_params, images)
        
        first_image = list(images.values())[0]
        _, image_height, image_width = first_image.shape
        
        number_of_gaussians = batch['initial_gaussians']['xyz'].shape[0]
        latent_code = nn.Parameter(torch.randn((1, self.latent_code_size), device=self.device).requires_grad_(True))
        
        test_time_optimizer = self.configure_test_time_optimizer(latent_code)
        test_progress_tqdm = tqdm(range(self.test_time_optimization_steps))
        for _ in test_progress_tqdm:
            latent_codes = latent_code.repeat(number_of_gaussians, 1)
            batch['initial_gaussians']['latent_codes'] = latent_codes
            predicted_gaussians = self.forward(initial_gaussians)
            gaussian_model.set_gaussian_properties(predicted_gaussians)
        
            rendered_images, rendered_depth = self.render_images(
                gaussian_model=gaussian_model,
                camera_parameters=input_camera_params,
                height=image_height,
                width=image_width,
                device=self.device
            )
        
            test_time_opt_loss = self.losses.compute_loss('val_opt', sequence_id, rendered_images, input_images, None, None, predicted_gaussians)
            test_time_opt_loss.backward()
            test_time_optimizer.step()
            test_progress_tqdm.set_postfix({'test loss': test_time_opt_loss.item()})
        
        test_time_optimizer.zero_grad()
        torch.set_grad_enabled(False)
        
        intrinsics = camera_params['intrinsics']
        id=sequence_id.replace('/EXP-1-head','')
        if self.current_epoch % 5 ==0:
            self.write_video('val',id,image_height,image_width,gaussian_model ,intrinsics ,self.background_color.to(self.device)  )
        
        if self.current_epoch== self.trainer.max_epochs-1:
            self.write_video('val',id,image_height,image_width,gaussian_model ,intrinsics ,self.background_color.to(self.device)  )
            print('saving gaussian_model as .ply')

        full_rendered_images, full_rendered_depth = self.render_images(
            gaussian_model=gaussian_model,
            camera_parameters=camera_params,
            height=image_height,
            width=image_width,
            device=self.device
        )
        full_loss = self.losses.compute_loss('val', sequence_id, full_rendered_images, images, full_rendered_depth, target_depths, predicted_gaussians)
        
        return full_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.gauss_predictor_model.parameters()},
            {'params': self.training_latent_codes.parameters(), 'lr': self.latent_code_learning_rate},
        ], lr=self.config['training']['lr'])
        return optimizer
    
    def configure_test_time_optimizer(self, latent_code):
        optimizer = torch.optim.Adam([latent_code], lr=self.latent_code_learning_rate)
        return optimizer