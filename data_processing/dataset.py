import random
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose, Intrinsics
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.trajectory import circle_around_axis
from dreifus.util.visualizer import ImageWindow
from dreifus.vector import Vec3
from PIL import Image
from gaussian_splatting.utils.sh_utils import RGB2SH
from image_feature_projection import ImageFeatureProjection
from flame.flame_model_creation import FlameModelCreation
import torch.nn.functional as F

from gaussian_splatting.utils.graphics_utils import BasicPointCloud
import cv2
from elias.util.io import resize_img
from pathlib import Path
import open3d as o3d

##pixelnerf encoder
from torchvision import transforms
from pixelnerf.helper import conf as conf_pixelnerf
import pixelnerf.encoder as encoder

class FaceDataset(Dataset):
    def __init__(self, config, data_split='train'):
        dataset_num_samples_train = config['data']['dataset_num_samples_train']
        repeat_training_samples_num = config['data']['repeat_training_samples_num']
        
        assert data_split in ['train', 'val', 'test'], f'data_split unknown: {data_split}'
        assert isinstance(dataset_num_samples_train, int) or dataset_num_samples_train == 'full'
        
        if data_split == 'test':
            print('Using test split!')
        
        self.config = config
        self.data_split = data_split
        self.downscale_factor = config['data']['downscale_factor']
        self.rescale_factor_intrinsics = config['data']['rescale_factor_intrinsics']
        self.apply_alpha_map = config['data']['apply_alpha_map']
        self.use_depth = config['data']['use_depth']
        self.use_monocular_depth = config['data']['use_monocular_depth']
        
        self.use_color_correction=config['data']['use_color_correction']
        self.mask_Torso=config['data']['mask_Torso']
        
        self.hair_gaussians_enabled=config['data'].get('hair_gaussians_enabled', False)

        self.force_rebuild = config['data']['force_rebuild']
        self.save_results = config['data'].get('save_results', True)
        self.encoder_type = config['data'].get('encoder', 'F')
        self.random_input_image_enabled = config['data'].get('random_input_image_enabled', False)
        print(f'ENCODER: {self.encoder_type}')
        
        with open(f'./data_splits/{data_split}.txt') as file:
            self.face_ids = [line.replace('\n', '') for line in file.readlines()]            
            print(f'{data_split} self.face_ids', self.face_ids)
        
        self.sequences = self.get_sequences_for_face_ids()
        
        if isinstance(dataset_num_samples_train, int) and self.data_split=='train':
            # if dataset_num_samples is an int then take only dataset_num_samples samples
            # useful for overfitting
            self.sequences = self.sequences[:dataset_num_samples_train]
            
        if repeat_training_samples_num > 1 and self.data_split=='train':
            self.sequences = self.sequences * repeat_training_samples_num
            
        self.sequence_cache = {}
        self.downscaledW=0
        self.downscaledH=0

        #Depth GT
        if self.use_monocular_depth: 
            #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            model_type ='MiDaS_small'
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            #self.midas.cuda()
            self.midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.dpt_transform
            
        self.rebuild_idx = [self.force_rebuild] * len(self.sequences)
        
    def get_sequences_for_face_ids(self):
        sequence_paths = []
        
        for face_id in self.face_ids:
            face_path = f"{self.config['data']['path']}/{face_id}"
            sequences_path = f"{face_path}/sequences"
            for sequence in os.listdir(sequences_path):
                sequence_path = f"{sequences_path}/{sequence}"
                timesteps_path = f"{sequence_path}/timesteps"
                for timestep in os.listdir(timesteps_path):
                    sequence_paths.append({
                        'camera_params_path': f"{face_path}/calibration/camera_params.json",
                        'flame_params_path': f"{sequence_path}/annotations/tracking/FLAME2023_v2/tracked_flame_params.npz",
                        #'flame_path': f"{sequence_path}/annotations/tracking/FLAME2023_v2/flame.obj",
                        #'gaussian_features_path': f"{timesteps_path}/{timestep}/gaussian_features.npy",
                        #'gaussian_colors_path': f"{timesteps_path}/{timestep}/gaussian_colors.npy",
                        'input_dictionary': f"{timesteps_path}/{timestep}/input_dictionary.npz",
                        'images_path': f"{timesteps_path}/{timestep}/images-2x",
                        'id': f"{face_id}/{sequence}"
                    })
                    
        return sequence_paths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        if not self.rebuild_idx[idx]:
            return torch.load(sequence['input_dictionary'])
        if not self.random_input_image_enabled:
            self.rebuild_idx[idx] = False
        
        print(f"id: {sequence['id']}")
        
        camera_params = self.get_camera_params(sequence)
        images = self.get_images(sequence, camera_params)
        depths = self.get_depths(sequence, camera_params)
        if self.use_monocular_depth:
         depths = self.get_monocular_depth(sequence, camera_params)

        
        pointcloud = self.get_flame_pointcloud(sequence)
        pointcloud, camera_params = self.scale_and_rotate(pointcloud, camera_params, sequence)
        
        input_serials = self.config['data']['input_serials']
        if self.random_input_image_enabled:
            input_serials = [random.choice(input_serials)]
        
        input_images = {serial: image for serial, image in images.items() if serial in input_serials}
        
        gaussian_colors = self.get_projected_colors(sequence, camera_params, pointcloud, input_images)
        gaussian_features = self.get_projected_features(sequence, camera_params, pointcloud, input_images)
        
        if self.hair_gaussians_enabled:
            pointcloud, gaussian_colors, gaussian_features = self.add_hair_gaussians(sequence, camera_params, pointcloud, input_images, gaussian_colors, gaussian_features)
        
        camera_params['cam_2_world_poses'] = FaceDataset.poses_to_dicts(camera_params['cam_2_world_poses'])
        
        input_dictionary = {
            'initial_gaussians': {
                'xyz': torch.tensor(pointcloud, dtype=torch.float32),
                'color': torch.tensor(gaussian_colors, dtype=torch.float32),
                'features': torch.tensor(gaussian_features, dtype=torch.float32),
                # Placeholders
                'scaling': torch.zeros([pointcloud.shape[0], 3], dtype=torch.float32),
                'rotation': torch.zeros([pointcloud.shape[0], 4], dtype=torch.float32),
                'opacity': torch.zeros([pointcloud.shape[0], 1], dtype=torch.float32),  
            },
            'camera_params': camera_params,
            'images': {serial: torch.tensor(image).permute(2, 0, 1) for serial, image in images.items()},
            'input_serials': input_serials,
            'input_images': {serial: torch.tensor(image).permute(2, 0, 1) for serial, image in input_images.items()},
            'depths': depths,
            'sequence_id': sequence['id']
        }
        print(f"no. of gaussians: {input_dictionary['initial_gaussians']['xyz'].shape[0]}")
        if self.save_results:
            torch.save(input_dictionary, sequence['input_dictionary'])
        return input_dictionary
    
    ##### IMAGES #####
    
    def get_camera_params(self, sequence):
        camera_params = json.load(open(sequence['camera_params_path'], 'r'))
        intrinsics = Intrinsics(camera_params['intrinsics'])  # Note: In the demo, all cameras have the same intrinsics
        intrinsics = intrinsics.rescale(float(self.rescale_factor_intrinsics)/float(self.downscale_factor))  # Note: when images are rescaled, intrinsics has to be scaled as well!
        cam_2_world_poses = dict()  # serial => world_2_cam_pose
        for serial, world_2_cam_pose in camera_params['world_2_cam'].items():
            world_2_cam_pose = Pose(world_2_cam_pose, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
            cam_2_world_pose = world_2_cam_pose.change_pose_type(PoseType.CAM_2_WORLD)
            cam_2_world_poses[serial] = cam_2_world_pose
            
        return {
            'intrinsics': intrinsics,
            'cam_2_world_poses': cam_2_world_poses
        }
            
    def get_depths(self, sequence, camera_params):
     depth_images=dict()
     intrinsics = Intrinsics(camera_params['intrinsics'])  # Note: In the demo, all cameras have the same intrinsics
     intrinsics = intrinsics.rescale(float(self.rescale_factor_intrinsics)/float(self.downscale_factor))  # Note: when images are rescaled, intrinsics has to be scaled as well!
     if self.use_depth:
        images_path = sequence['images_path']
        for serial in camera_params['cam_2_world_poses']:
            image_path = f"{images_path}/cam_{serial}.jpg"
            image = Image.open(image_path)
            image = image.resize((int(image.width / self.downscale_factor), int(image.height / self.downscale_factor)))  # Resize image
            image = np.array(image)
            image_height, image_width,_ =image.shape
            
            pcd_path=str(image_path).replace(f'/images-2x/cam_{serial}.jpg', '/colmap/pointclouds/pointcloud_16.pcd')
            pcd = o3d.io.read_point_cloud(pcd_path)
            points=np.array(pcd.points)
            points = np.concatenate((points, np.ones([points.shape[0], 1])), axis=1)
            
            pose=camera_params['cam_2_world_poses'][serial]
            p_cams = pose @ points.T #(4, 1185406)
            depth=p_cams[2,:]
            p_screens = p_cams / p_cams[2,:]
            p_screens = p_screens[0:3,:]

            p_screens = intrinsics @ p_screens
            p_screens = p_screens.T

            # Initialize the depth map with a high value (inf for example)
            depth_map = np.full((image_height, image_width), np.inf)

            # Fill the depth map with the depth values
            for i in range(p_screens.shape[0]):
                    x, y = int(p_screens[i, 0]), int(p_screens[i, 1])
                    z = depth[i]
                    if 0 <= x < image_width and 0 <= y < image_height:
                        if depth_map[y, x] == np.inf : #or z < depth_map[y, x]:  # Only update if the new point is closer
                            depth_map[y, x] = z

            # Replace inf values with zero or some large number indicating no data
            depth_map[depth_map == np.inf] = 0
            depth_images[serial]=depth_map
     return depth_images
    

    def get_monocular_depth(self, sequence, camera_params):
     depth_images=dict()
     if self.use_depth:
        images_path = sequence['images_path']
        for serial in camera_params['cam_2_world_poses']:
            image_path = f"{images_path}/cam_{serial}.jpg"
            image = Image.open(image_path)
            image = image.resize((int(image.width / self.downscale_factor), int(image.height / self.downscale_factor)))  # Resize image
            image = np.array(image)
            
            ## GT depth maps colmap
            # depth_map_file = str(image_path).replace("/images-2x/cam_", "/colmap/depth_maps_geometric/16/cam_").replace("jpg", "png")
            # depth_image = Image.open(depth_map_file)
            # depth_image = depth_image.resize((int(W), int(H)))
            # depth_array = np.array(depth_image)
            # depth_images[serial]=depth_array
            
            #GT depth maps midas
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_batch = self.midas_transform(img)#.to(device)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_images[serial]=prediction
     return depth_images


    def get_images(self, sequence, camera_params):

        # Load images
        images = dict()  # serial => image
        #depth_images=dict()
        images_path = sequence['images_path']
        removed_serials=[]
        for serial in camera_params['cam_2_world_poses']:
            image_path = f"{images_path}/cam_{serial}.jpg"
            image = Image.open(image_path)
            
            # Resize image
            W,H=image.width / self.downscale_factor, image.height / self.downscale_factor
            self.downscaledW=W
            self.downscaledH=H
            image = image.resize((int(image.width / self.downscale_factor), int(image.height / self.downscale_factor)))  # Resize image
            image = np.array(image) 
        
            #color_correction
            if self.use_color_correction: 
                color_correction= str(image_path).replace('/timesteps/frame_00000/images-2x/', '/annotations/color_correction/').replace('.jpg', '.npy')
                color_correction_path=color_correction.replace('cam_','')
                image = image / 255.0
                affine_color_transform = np.load(f'{color_correction_path}')
                image = image @ affine_color_transform[:3, :3] + affine_color_transform[np.newaxis, :3, 3]
                image = np.clip(image, 0, 1)
            else:
                image = np.array(image) / 255
            
            # apply alpha map
            if self.apply_alpha_map:
                alpha_map = Image.open(str(image_path).replace('/images-2x', '/alpha_map/').replace('.jpg', '.png'))
                alpha_map = alpha_map.resize((int(image.shape[1]), int(image.shape[0]))) # using image is important, as they might not have same initial size
                alpha_map = (np.array(alpha_map) / 255).astype(np.bool_)
                image[~alpha_map,:] = 1
            
            elif self.mask_Torso:
                segmentation_path= image_path.replace('/images-2x', '/facer_segmentation_masks').replace('.jpg', '.png').replace('cam_','color_segmentation_cam_')
                if not Path(segmentation_path).exists():
                    print('!!!!! SEG Mask Dosent EXIST for:',serial)
                    #del camera_params['cam_2_world_poses'][serial]
                    removed_serials.append(serial)
                else:
                    segmentation_mask=np.array(Image.open(segmentation_path).convert('L'))
                    segmentation_mask = resize_img(segmentation_mask,
                                                        (image.shape[1] / segmentation_mask.shape[1],
                                                        image.shape[0] / segmentation_mask.shape[0]),
                                                        interpolation='nearest')
                    remove_ids =[0,128,92]  # torso , background
                    torso_mask = np.zeros_like(segmentation_mask, dtype=bool)
                    for remove_id in remove_ids:
                        torso_mask = torso_mask | (segmentation_mask == remove_id)
                    image[torso_mask] = 1
                    images[serial] = image
           
            ## GT depth maps colmap
            # depth_map_file = str(image_path).replace("/images-2x/cam_", "/colmap/depth_maps_geometric/16/cam_").replace("jpg", "png")
            # depth_image = Image.open(depth_map_file)
            # depth_image = depth_image.resize((int(W), int(H)))
            # depth_array = np.array(depth_image)
            # depth_images[serial]=depth_array

        if len(removed_serials)>0:
            for serial in removed_serials:
             del camera_params['cam_2_world_poses'][serial]
        return images
    
    ##### POINT CLOUD #####
            
    def get_flame_pointcloud(self, sequence):
        flame_vertices = FlameModelCreation.create_flame_model(sequence['flame_params_path'], self.config)
        return flame_vertices
    
    #### NORMALIZE POINT CLOUD AND CAMERAS ####
                
    def scale_and_rotate(self, pointcloud, camera_params, sequence):
        # if self.scale_and_rotation_mode == 'flame':
        #     model_transformations = FlameModelCreation.get_transformation_and_scale(sequence['flame_params_path'])
        #     pointcloud = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], pointcloud.shape[1], 1))], dim=-1)
        #     pointcloud = model_transformations.double().permute(0, 2, 1) @ pointcloud
        #     flame_vertices = flame_vertices[..., :3]
        #     pointcloud = model_transformations @ pointcloud
        # elif self.scale_and_rotation_mode == 'cameras':
        model_transformations = FlameModelCreation.get_transformation_and_scale(sequence['flame_params_path'])
        
        cam_2_world_poses = camera_params['cam_2_world_poses']
        for serial, cam_2_world_pose in cam_2_world_poses.items():
            new_pose = np.linalg.inv(model_transformations) @ np.array(cam_2_world_pose)
            cam_2_world_poses[serial] = Pose(new_pose, 
                                            camera_coordinate_convention=cam_2_world_pose.camera_coordinate_convention,
                                            pose_type=cam_2_world_pose.pose_type,
                                            disable_rotation_check=True
                                            )
        return pointcloud, camera_params
    
    ##### FARL FEATURES #####
    
    def get_projected_colors(self, sequence, camera_params, pointcloud, images, max_dist_factor=0.5):
        gaussian_colors = self.project_features_onto_gaussians(
            camera_params, pointcloud, images, images, max_dist_factor=max_dist_factor
        )
        gaussian_colors = RGB2SH(gaussian_colors)
        return gaussian_colors
    
    def get_projected_features(self, sequence, camera_params, pointcloud, images, max_dist_factor=1):
        if self.encoder_type=='N':
            # None
            gaussian_features = np.zeros([pointcloud.shape[0], 1], dtype=np.float32)
        elif self.encoder_type=='R':
            # ResNet
            gaussian_features = self.get_projected_resnet_features(sequence, camera_params, pointcloud, images, max_dist_factor=max_dist_factor)
        elif self.encoder_type=='F':
            # FaRL
            gaussian_features = self.get_projected_farl_features(sequence, camera_params, pointcloud, images, max_dist_factor=max_dist_factor)
        else:
            raise NotImplementedError(f'Unknwon image encoder {self.encoder_type}')
        
        return gaussian_features
    
    def get_projected_farl_features(self, sequence, camera_params, pointcloud, images, max_dist_factor=0.5):        
        farl_features = self.get_farl_features(sequence)
        gaussian_features = self.project_features_onto_gaussians(
            camera_params, pointcloud, farl_features, images, max_dist_factor=max_dist_factor
        )
        return gaussian_features
    
    def get_projected_resnet_features(self, sequence, camera_params, pointcloud, images, max_dist_factor=0.5):
        features = self.get_resnet_features(sequence)
        gaussian_features = self.project_features_onto_gaussians(
            camera_params, pointcloud, features, images, max_dist_factor=max_dist_factor
        )
        return gaussian_features
    
    def get_farl_features(self, sequence):
        image_features = dict()
        
        images_path = sequence['images_path']
        
        for serial in self.config['data']['input_serials']:
            farl_path = f"{images_path}/cam_{serial}_farl.npy"
            features_for_image = np.load(farl_path)
            inv_grid_path = f"{images_path}/cam_{serial}_inv_grid.npy"
            inv_grid = np.load(inv_grid_path)
            features_for_image = F.grid_sample(torch.tensor(features_for_image), torch.tensor(inv_grid), mode='bilinear', align_corners=False)
            features_for_image = features_for_image.squeeze(0).permute(1,2,0).numpy()
            image_features[serial] = features_for_image
            
        return image_features
    
    def make_encoder(self,conf, **kwargs):
     net = encoder.SpatialEncoder.from_conf(conf, **kwargs)
     return net

    def get_resnet_features(self, sequence):
        H,W=self.downscaledH, self.downscaledW
        transform = transforms.Compose([
            transforms.Resize((int(H),int(W))),  # Resize images to the desired size (H, W)
            transforms.ToTensor()  # Convert PIL Image to Tensor
            ])

        image_tensors = []
        images_path = sequence['images_path']
        for serial in self.config['data']['input_serials']:
            image_path = f"{images_path}/cam_{serial}.jpg"
            image = Image.open(image_path)
            image = image.resize((int(W), int(H)))
            image_tensor = transform(image)
            image_tensors.append(image_tensor)

        batch_images = torch.stack(image_tensors)
        encoder = self.make_encoder(conf_pixelnerf["model"]["encoder"])
        with torch.no_grad():
          encoder.eval().cpu()
          latent=encoder(batch_images)
        
        image_features = dict()
        images_path = sequence['images_path']
        i=0
        for serial in self.config['data']['input_serials']:
            image_features[serial] = latent[i].permute(1,2,0).numpy() #detach().cpu().numpy()
            i+=1
        return image_features
    
            
    def project_features_onto_gaussians(self, camera_params, pointcloud, farl_features, images, max_dist_factor=0.5):    
        first_image = list(images.values())[0]
        
        gaussian_features = ImageFeatureProjection.project_features_onto_gaussians(
            xyz=pointcloud,
            features_per_image=farl_features,
            cam_2_world_poses=camera_params['cam_2_world_poses'],
            intrinsics=camera_params['intrinsics'],
            original_image_height=first_image.shape[0],
            original_image_width=first_image.shape[1],
            max_dist_factor=max_dist_factor
        )
        return gaussian_features
    
    def add_hair_gaussians(self, sequence, camera_params, pointcloud, images, gaussian_colors, gaussian_features):
        try:
            not_hair_indicator = gaussian_features[:,-10:-1].sum(-1)
            hair_pointcloud = pointcloud[not_hair_indicator<0.5].copy()    
            
            hair_pointcloud = np.concatenate([
                hair_pointcloud * 1.1,
                hair_pointcloud * 1.125,
                hair_pointcloud * 1.14
            ], axis=0)
            
            if hair_pointcloud.shape[0] > 15_000:
                # Because of memory issues, allow max 30k extra points
                hair_pointcloud = hair_pointcloud[torch.randperm(hair_pointcloud.shape[0])][:15_000]
            
            hair_gaussian_colors = self.get_projected_colors(sequence, camera_params, hair_pointcloud, images)
            hair_gaussian_features = self.get_projected_farl_features(sequence, camera_params, hair_pointcloud, images)
            
            inner_flame_features = np.ones([pointcloud.shape[0], 1])
            outer_flame_features = np.zeros([hair_pointcloud.shape[0], 1])
            inner_vs_outer_flame_features = np.concatenate([inner_flame_features, outer_flame_features], axis=0)
            
            pointcloud = np.concatenate([pointcloud, hair_pointcloud], axis=0)
            gaussian_colors = np.concatenate([gaussian_colors, hair_gaussian_colors], axis=0)
            gaussian_features = np.concatenate([gaussian_features, hair_gaussian_features], axis=0)
            gaussian_features = np.concatenate([gaussian_features, inner_vs_outer_flame_features], axis=-1)
        except:
            print(f'[WARNING] failed adding hair gaussians for {sequence["id"]}')
            inner_flame_features = np.ones([pointcloud.shape[0], 1])
            gaussian_features = np.concatenate([gaussian_features, inner_flame_features], axis=-1)
            return pointcloud, gaussian_colors, gaussian_features    
        
        return pointcloud, gaussian_colors, gaussian_features
    
    @classmethod
    def poses_to_dicts(cls, poses):
        # needed as otherwise pose is not accessible in pytorch lightning module
        # probably some serialization problem?
        pose_dict = {}
        for serial, pose in poses.items():
            pose_dict[serial] = {
                'matrix_or_rotation': np.array(pose),
                'pose_type': pose.pose_type,
                'camera_coordinate_convention': pose.camera_coordinate_convention
            }
        
        return pose_dict
            
    @classmethod
    def dicts_to_poses(cls, poses):
        pose_dict = {}
        for serial, pose in poses.items():
            pose_dict[serial] = Pose(**pose)
        
        return pose_dict