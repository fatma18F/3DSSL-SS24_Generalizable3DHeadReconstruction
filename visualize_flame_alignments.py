from flame.flame_model_creation import FlameModelCreation
from data_processing.dataset import FaceDataset
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
import torch
import k3d
import yaml
import numpy as np
import os

os.environ["DISPLAY"] = ":10.0"

face_dataset = FaceDataset(yaml.safe_load(open('./configs/settings.yaml', 'r')))

def get_aligned_pointcloud(flame_params_path):
    vertices = FlameModelCreation.create_flame_model(flame_params_path)
    pointlcoud = BasicPointCloud(vertices, torch.randn((len(vertices), 3)), None)
    
    #new_pointcloud = FaceDataset.normalize_flame_and_cameras(None, pointlcoud, None)
    
    return pointlcoud

def get_images(flame_params_path, camera_params):
    image_path = flame_params_path[:flame_params_path.find('/annotations/')]
    image_path += f"/timesteps/frame_00000/images-2x"
    return FaceDataset.get_images(face_dataset, {'images_path': image_path}, camera_params)

def get_translated_camera_params(flame_params_path):
    camera_params_path = flame_params_path[:flame_params_path.find('/sequences/')]
    camera_params_path += f"/calibration/camera_params.json"
    camera_params = FaceDataset.get_camera_params(face_dataset, {'camera_params_path': camera_params_path})
    FaceDataset.scale_and_rotate(face_dataset, camera_params, {'flame_params_path': flame_params_path})
    
    return camera_params

p1_path = '/home/antonio/projects/3dssl/3DSSL-SS24_Generalizable3DHeadReconstruction/data/017/sequences/EMO-1-shout+laugh/annotations/tracking/FLAME2023_v2/tracked_flame_params.npz'
p2_path = '/home/antonio/projects/3dssl/3DSSL-SS24_Generalizable3DHeadReconstruction/data/033/sequences/SEN-10-port_strong_smokey/annotations/tracking/FLAME2023_v2/tracked_flame_params.npz'
p3_path = '/home/antonio/projects/3dssl/3DSSL-SS24_Generalizable3DHeadReconstruction/data/058/sequences/SEN-10-port_strong_smokey/annotations/tracking/FLAME2023_v2/tracked_flame_params.npz'
p4_path = '/home/antonio/projects/3dssl/3DSSL-SS24_Generalizable3DHeadReconstruction/data/067/sequences/SEN-10-port_strong_smokey/annotations/tracking/FLAME2023_v2/tracked_flame_params.npz'

p1_pointcloud = get_aligned_pointcloud(p1_path)
p2_pointcloud = get_aligned_pointcloud(p2_path)
p3_pointcloud = get_aligned_pointcloud(p3_path)
p4_pointcloud = get_aligned_pointcloud(p4_path)

plot = k3d.plot()

plot += k3d.points(p1_pointcloud.points, point_size=0.01, color=0xff0000)
plot += k3d.points(p2_pointcloud.points, point_size=0.01, color=0x00ff00)
plot += k3d.points(p3_pointcloud.points, point_size=0.01, color=0x0000ff)
plot += k3d.points(p4_pointcloud.points, point_size=0.01, color=0x000000)

plot.display()

import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum

p1_camera_params = get_translated_camera_params(p1_path)
p1_images = get_images(p1_path, p1_camera_params)

p2_camera_params = get_translated_camera_params(p2_path)
p2_images = get_images(p2_path, p2_camera_params)


p = pv.Plotter()
#pv.start_xvfb()

#add_coordinate_axes(p)

for serial, image in p1_images.items():
    add_camera_frustum(p, p1_camera_params['cam_2_world_poses'][serial], p1_camera_params['intrinsics'], image=image)
p.add_points(np.array(p1_pointcloud.points))

for serial, image in p2_images.items():
    add_camera_frustum(p, p2_camera_params['cam_2_world_poses'][serial], p2_camera_params['intrinsics'], image=image)
p.add_points(np.array(p2_pointcloud.points))

p.show()