import pytorch_lightning as L
from data_processing.dataset import FaceDataset
from model.model_collection import ModelCollection
from gaussian_splatting.scene.mlp_gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.arguments import OptimizationParams, PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.viewer import GaussianViewer
import torch
import time
from tqdm import tqdm
from dreifus.image import Img
import lpips
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.regression import pearson_corrcoef
from dreifus.vector import Vec3

from dreifus.trajectory import circle_around_axis
from mediapy import VideoWriter
import matplotlib.pyplot as plt 
import os
from losses.losses import Losses

from dreifus.camera import CameraCoordinateConvention

class GaussPredictor(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        self.config = config
        self.background_color = torch.tensor(self.config['render']['background_color'], dtype=torch.float32)
        
        self.gauss_predictor_model = ModelCollection.get_model(config)
        self.gaussian_models = {}
        
        # For visualization if enabled
        self.viewer_enabled = self.config['visualization']['enabled']
        self.use_depth=self.config['model']['use_depth']
        self.viewers = {}
        
        with open(f'./data_splits/val.txt') as file:
            self.test_ids = [line.replace('\n', '') for line in file.readlines()]
            
        self.losses = Losses(config=config, gauss_predictor=self)
        
    def forward(self, initial_gaussians):
        return self.gauss_predictor_model(initial_gaussians)
        
    def training_step(self, batch, batch_idx):
        initial_gaussians = batch['initial_gaussians']
        images = batch['images']
        target_depths = batch['depths']
        sequence_id = batch['sequence_id']
        
        camera_params = batch['camera_params']
        camera_params['cam_2_world_poses'] = FaceDataset.dicts_to_poses(camera_params['cam_2_world_poses'])
        
        input_serials = batch['input_serials']
        input_cam_2_world_poses = {serial: cam_2_world_pose for serial, cam_2_world_pose in camera_params['cam_2_world_poses'].items() if serial in input_serials}
        input_camera_params = {'cam_2_world_poses': input_cam_2_world_poses, 'intrinsics': camera_params['intrinsics']}
        
        gaussian_model = self.get_gaussian_model(sequence_id)
        id=sequence_id.replace('/EXP-1-head','')
        
        assert id not in self.test_ids
        
        #if int(id)>70 and int(id)<100 :
        self.visualize(sequence_id, input_camera_params, images)
        
        self.max_scale, self.max_displacement= self.define_thresholds(initial_gaussians)
        predicted_gaussians = self.forward(initial_gaussians)
        gaussian_model.set_gaussian_properties(predicted_gaussians)
        
        first_image = list(images.values())[0]
        _, image_height, image_width = first_image.shape
        
        rendered_images, rendered_depth = self.render_images(
            gaussian_model=gaussian_model,
            camera_parameters=camera_params,
            height=image_height,
            width=image_width,
            device=self.device
        )
        
        loss = self.losses.compute_loss('train', sequence_id, rendered_images, images, rendered_depth, target_depths, predicted_gaussians)
        
        # if self.current_epoch%10==0 and id == '057':
        #  print('saving gaussian_model as .ply')
        #  gaussian_model.save_ply(f"/home/ayed/main/output/PCD/mlp_GS_{id}_{self.current_epoch}.ply")
       
        #for debugging
        #if self.current_epoch % 10 ==0 and id in [ '194','038','328','323', '256','255','251']:
        if self.current_epoch % 10 == 0 and id in [ '328','323', '256','255','251']:
            intrinsics = camera_params['intrinsics']
            self.write_video('train' ,id,image_height,image_width,gaussian_model ,intrinsics ,self.background_color.to(self.device))
            
        if self.current_epoch == 0 and self.global_step==0:
            self.log('misc/number_of_gaussians', initial_gaussians['xyz'].shape[0], batch_size=1)
         
        return loss
    
    def validation_step(self, batch, batch_idx):
        initial_gaussians = batch['initial_gaussians']
        images = batch['images']
        target_depths = batch['depths']
        sequence_id = batch['sequence_id']
        
        camera_params = batch['camera_params']
        camera_params['cam_2_world_poses'] = FaceDataset.dicts_to_poses(camera_params['cam_2_world_poses'])
        
        input_serials = batch['input_serials']
        input_cam_2_world_poses = {serial: cam_2_world_pose for serial, cam_2_world_pose in camera_params['cam_2_world_poses'].items() if serial in input_serials}
        input_camera_params = {'cam_2_world_poses': input_cam_2_world_poses, 'intrinsics': camera_params['intrinsics']}
        
        gaussian_model = self.get_gaussian_model(sequence_id)
        self.visualize(sequence_id, input_camera_params, images)
        
        if self.config['model']['name']=='gauss_predictor_mlp_improved':
            self.max_scale,self.max_displacement =self.define_thresholds(initial_gaussians)
        predicted_gaussians = self.forward(initial_gaussians)
        gaussian_model.set_gaussian_properties(predicted_gaussians)
        
        first_image = list(images.values())[0]
        _, image_height, image_width = first_image.shape
        
        rendered_images, rendered_depth = self.render_images(
            gaussian_model=gaussian_model,
            camera_parameters=camera_params,
            height=image_height,
            width=image_width,
            device=self.device
        )
        
        loss = self.losses.compute_loss('val', sequence_id, rendered_images, images, rendered_depth, target_depths, predicted_gaussians)
        
        #if self.current_epoch % 5 ==0:
        intrinsics = camera_params['intrinsics']
        id=sequence_id.replace('/EXP-1-head','')
        
        assert id in self.test_ids
        
        if self.current_epoch % 5 == 0 or self.current_epoch == self.trainer.max_epochs-1:
            self.write_video('val', id, image_height, image_width, gaussian_model, intrinsics, self.background_color.to(self.device))
        return loss
        
    def write_video(self,mode,id,H ,W ,gaussian_model, intrinsics, bg_color):
        # Generate video
        fps = 24
        seconds = 4
        trajectory = circle_around_axis(seconds * fps,
                                    axis=Vec3(0, 0, -1),
                                    up=Vec3(0, -1, 0),
                                    move=Vec3(0, 0, 1),
                                    distance=0.3)

        path = f"output/{self.logger.experiment.id}"
        if not os.path.exists(path):
            os.makedirs(path)
        with VideoWriter(f"{path}/{mode}_person_{int(id):03}_{self.current_epoch:03}.mp4", (H, W), fps=fps) as video_writer:
         for pose in tqdm(trajectory, desc="Generating video"):
            gs_camera = pose_to_rendercam(pose, intrinsics, W, H)
            output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color)
            rendered_image = output['render']
            rendered_image = Img.from_torch(rendered_image.flip([1,2])).to_numpy().img
            video_writer.add_image(rendered_image)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gauss_predictor_model.parameters(), lr=self.config['training']['lr'], weight_decay=1e-5)

        return optimizer
    
    def render_images(self, gaussian_model, camera_parameters, height, width, device):
        intrinsics = camera_parameters['intrinsics']
        cam_2_world_poses = camera_parameters['cam_2_world_poses']
        
        rendered_images = {}
        rendered_depth=0
        for serial, cam_2_world_pose in cam_2_world_poses.items():
            gs_camera = pose_to_rendercam(cam_2_world_pose, intrinsics, width, height, device=device)
            output = render(gs_camera, gaussian_model, PipelineParams2(), self.background_color.to(device), return_depth=self.use_depth)
            rendered_images[serial] = output['render']
            if self.use_depth:
             rendered_depth = output["depth"][0]

        return rendered_images,rendered_depth
            
    def get_gaussian_model(self, sequence_id):
        #return GaussianModel(0)
        if sequence_id not in self.gaussian_models:
            self.gaussian_models[sequence_id] = GaussianModel(0) # properties are replaced each step
        return self.gaussian_models[sequence_id]
            
    def visualize(self, sequence_id, camera_parameters, images):
        if self.viewer_enabled and sequence_id not in self.viewers:
            cam_2_world_poses = camera_parameters['cam_2_world_poses']
            intrinsics = camera_parameters['intrinsics']
            
            viewer_poses = []
            viewer_intrinsics = []
            viewer_images = []
            for serial, cam_2_world_pose in cam_2_world_poses.items():
                viewer_poses.append(cam_2_world_pose)
                viewer_intrinsics.append(intrinsics)
                viewer_images.append((images[serial]).permute(1,2,0).cpu().numpy())
                
            gaussian_viewer = GaussianViewer(self.get_gaussian_model(sequence_id),
                                         poses=viewer_poses,
                                         viewer_port=8000+int(sequence_id[:sequence_id.find('/')]),
                                         intrinsics=viewer_intrinsics,
                                         images=viewer_images,
                                         size=0.03)
                
            self.viewers[sequence_id] = gaussian_viewer
            gaussian_viewer.server.set_up_direction("+y")

    def define_thresholds(self,input):
        # Assuming initial_scales and initial_positions are available
        max_scale = 0#mean_scale + 2 * std_scale

        initial_positions = input['xyz']
        mean_displacement = torch.mean(initial_positions, dim=0)
        std_displacement = torch.std(initial_positions, dim=0)
        max_displacement = torch.norm(mean_displacement + 2 * std_displacement)
        
        return max_scale, max_displacement