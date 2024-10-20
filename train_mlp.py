import json
from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from PIL import Image
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose, Intrinsics
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.trajectory import circle_around_axis
from dreifus.util.visualizer import ImageWindow
from dreifus.vector import Vec3
from gaussian_splatting.arguments import OptimizationParams, PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.mlp_gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.viewer import GaussianViewer
from mediapy import VideoWriter
from tqdm import tqdm
import time
from image_feature_projection import ImageFeatureProjection
from mlp.gauss_predictor_mlp import GaussPredictorMLP


from torchmetrics.functional.regression import pearson_corrcoef
from image_feature_projection import ImageFeatureProjection
from mlp.gauss_predictor_mlp import GaussPredictorMLP
from loss import l1_loss, ssim ,lambda_dssim
import cv2

##pixelnerf encoder
from torchvision import transforms
from mlp.pixelnerf.helper import conf
import mlp.pixelnerf.encoder as encoder
def make_encoder(conf, **kwargs):
    net = encoder.SpatialEncoder.from_conf(conf, **kwargs)
    return net 



if __name__ == '__main__':
    IMAGE_CONFIGURATION = 'full'#three_views' 
    use_visualization_window = True
    use_viewer = True
    n_iterations = 5_000
    downscale_factor = 4  # By how much to downscale the images
    tracker2='EMO-1-shout+laugh'
    tracker1='SEN-10-port_strong_smokey'

    PERSON='017'
    FLAME_PATH=f'FLAME_output/Flame_{PERSON}_30k.obj'
    #FLAME_PATH='/home/ayed/mlp/FLAME_output/obj_mesh_50k_point.obj'
    TRACKER=tracker2

    print(f'Running Gaussian Splatting for image configuration: {IMAGE_CONFIGURATION} for Person {PERSON}')


    #pixelnerf encoder
    path=f'/home/ayed/Flame_fittings/Ryoto-single-timestep/000/sequences/EMO-1-shout+laugh/timesteps/frame_00000/images-2x/222200039.jpg'
    image = Image.open(path)
    image = image.resize((int(image.width / downscale_factor), int(image.height / downscale_factor)))  # Resize image
    H,W,_= np.array(image).shape
    image_tensors = []
    transform = transforms.Compose([
    transforms.Resize((H, W)),  # Resize images to the desired size (H, W)
    transforms.ToTensor()  # Convert PIL Image to Tensor
    ])

    #Depth
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    # Use transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.dpt_transform
    # ==========================================================
    # Data
    # ==========================================================
    print(f'Running Gaussian Splatting for Person: {PERSON}')

    # Load camera poses and intrinsics
    #camera_params = json.load(open('data/camera_params.json'))
    camera_params = json.load(open(f'/home/ayed/Flame_fittings/Ryoto-single-timestep/{PERSON}/calibration/camera_params.json'))
    intrinsics = Intrinsics(camera_params['intrinsics'])  # Note: In the demo, all cameras have the same intrinsics
    intrinsics = intrinsics.rescale(1.0 /( downscale_factor*2.0))  # Note: when images are rescaled, intrinsics has to be scaled as well!
    cam_2_world_poses = dict()  # serial => world_2_cam_pose
    for serial, cam_2_world_pose in camera_params['world_2_cam'].items():
        cam_2_world_pose = Pose(cam_2_world_pose, pose_type=PoseType.WORLD_2_CAM)
        cam_2_world_pose.change_pose_type(PoseType.CAM_2_WORLD)
        cam_2_world_poses[f'cam_{serial}'] = cam_2_world_pose

    
    # Load images
    images = dict()  # serial => image
    depth_images = dict()

    serials = []
    for image_file in Path(f'/home/ayed/Flame_fittings/Ryoto-single-timestep/{PERSON}/sequences/{TRACKER}/timesteps/frame_00000/images-2x').iterdir():
        serial = image_file.stem
        image = Image.open(image_file)

        alpha_map_file = str(image_file).replace("/images-2x/", "/alpha_map/").replace("jpg", "png")
        alpha_map = Image.open(alpha_map_file)
        alpha_map = alpha_map.resize((int(image.width / downscale_factor), int(image.height / downscale_factor)))  # Resize image
        alpha_map = np.array(alpha_map)
        mask = ~alpha_map.astype(bool)

        W,H=image.width / downscale_factor, image.height / downscale_factor
        image = image.resize((int(W), int(H)))  # Resize image
        #if serial in ['221501007', '222200037', '222200046']:
        image_tensor = transform(image)
        image_tensors.append(image_tensor)
        #mask * image + (1 - mask)

        image = np.array(image)
        image[mask] = [255, 255, 255]   

        images[serial] = image
        serials.append(serial)

        
        img = cv2.imread(str(image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = midas_transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_images[serial]=prediction

        
    image_features = dict()
    batch_images = torch.stack(image_tensors)
    print('batch_images',batch_images.shape)#16, 3, 802, 550

    encoder = make_encoder(conf["model"]["encoder"])
    with torch.no_grad():
        encoder.eval()
        latent=encoder(batch_images)
    print('latent',latent.shape)#[16, 512, 802, 550]
    image_features = dict()
    i=0
    for serial in serials:
        image_features[serial] = latent[i].permute(1,2,0).detach().cpu().numpy()
        #feat = np.load(f'{os.getcwd()}/FaRL/outputs/{PERSON}/farl_{serial}.npy')
        #image_features[serial] = feat

        ##if serial in ['221501007', '222200037', '222200046']:
        i+=1
        
    # ==========================================================
    # Visualize
    # ==========================================================

    # Visualize camera poses and images
    print('Visualize camera poses and images')
    pv.start_xvfb()
    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    for serial in serials:
        add_camera_frustum(p, cam_2_world_poses[serial], intrinsics, image=images[serial])
    #p.show()

    # ==========================================================
    # Model
    # ==========================================================
    print('Model')
    # Setup 3D Gaussian Model and optimizer
    gaussian_model = GaussianModel(sh_degree=0)
    device = torch.device('cuda')
    bg_color = torch.ones(3, device=device, dtype=torch.float32)  # Use white background for rendering
    pointcloud = BasicPointCloud.from_obj(FLAME_PATH)# obj_mesh_50k_point.
    #pointcloud = BasicPointCloud(points=torch.randn((5000, 3)) / 20, colors=torch.randn((5000, 3)), normals=None)
    gauss_predictor_mlp = GaussPredictorMLP(input_image_features_dim=list(image_features.values())[0].shape[-1]).to('cuda')

    
    # Setup optimizer
    gaussian_model = GaussianModel(sh_degree=0)
    visualized_gaussian_model = GaussianModel(sh_degree=0)
    parser = ArgumentParser()
    params_opt = OptimizationParams(parser)
    optimizer = torch.optim.Adam(
        params=gauss_predictor_mlp.parameters(),
        lr=0.0001
    )
    
    # ==========================================================
    # Viewer
    # ==========================================================
    print('Viewer')
    # Setup interactive web-based 3D Viewer for 3D Gaussians
    if use_viewer:
        viewer_poses = [cam_2_world_poses[serial] for serial in serials]
        viewer_intrinsics = [intrinsics for _ in serials]
        viewer_images = [images[serial] for serial in serials]
        viewer = GaussianViewer(gaussian_model,
                                poses=viewer_poses,
                                intrinsics=viewer_intrinsics,
                                images=viewer_images,
                                size=0.03)
        viewer.server.set_up_direction("-y")

    # ==========================================================
    # Optimization Loop
    # ==========================================================
    print('Optimization Loop')
    H, W, _ = images[serials[0]].shape
    images = {serial: image / 255 for serial, image in images.items()}
    torch_images = {serial: torch.tensor(image, device=device).permute(2,0,1) for serial, image in images.items()}
    progress = tqdm(range(n_iterations))
    
    gaussian_model.create_from_pcd(pointcloud, spatial_lr_scale=1)
    input = gaussian_model.get_gaussian_properties()
    
    gaussian_features = ImageFeatureProjection.project_features_onto_gaussians(
        gaussian_model=gaussian_model, 
        features_per_image=image_features,
        cam_2_world_poses=cam_2_world_poses,
        intrinsics=intrinsics,
        image_height=H,
        image_width=W
    )
    gaussian_colors = ImageFeatureProjection.project_features_onto_gaussians(
        gaussian_model=gaussian_model, 
        features_per_image=images, # TODO: ImageFeatureProjection should work in torch
        cam_2_world_poses=cam_2_world_poses,
        intrinsics=intrinsics,
        image_height=H,
        image_width=W
    )
    input['features'] = torch.tensor(gaussian_features, dtype=torch.float32).to('cuda')
    input['color'] = torch.tensor(gaussian_colors, dtype=torch.float32).unsqueeze(1).to('cuda')
    
    for iteration in progress:
        res = gauss_predictor_mlp.forward({key: val.clone() for key, val in input.items()})
        gaussian_model.set_gaussian_properties(res)
        
        loss = torch.tensor(0, dtype=torch.float64).to('cuda')
        for serial in serials:
            #serial = serials[iteration % len(serials)]
            gt_image = torch_images[serial]
            cam_2_world_pose = cam_2_world_poses[serial]
            gs_camera = pose_to_rendercam(cam_2_world_pose, intrinsics, W, H)
            output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color,return_depth=True)
            rendered_image = output['render']
            # Loss
            #loss += 
            Ll1=(rendered_image - gt_image).abs().mean()
            loss += (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
            los= (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
            #print('L1 loss', los)
            loss+= los
            # with torch.no_grad():
            #  input_batch = midas_transform(rendered_image.permute(1,2,0).detach().cpu().numpy()).to(device)
            #  prediction = midas(input_batch)
            #  prediction = torch.nn.functional.interpolate(
            #     prediction.unsqueeze(1),
            #     size=image.shape[:2],
            #     mode="bicubic",
            #     align_corners=False,
            # ).squeeze()
            # print('prediction:',prediction.shape)
            
            # rendered_depth=prediction
            rendered_depth = output["depth"][0]

            #print('rendered_depth',rendered_depth.shape) 
            midas_depth = depth_images[serial].cuda()
            #print('midas_depth',midas_depth.shape) 

            rendered_depth = rendered_depth.reshape(-1, 1)
            midas_depth = midas_depth.reshape(-1, 1)

            depth_loss = min(
                            (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
            )
            depth_weight = 0.5
            loss += depth_weight * depth_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix({"loss": loss.item()})
        
    # Generate video
    fps = 24
    seconds = 4
    trajectory = circle_around_axis(seconds * fps,
                                    axis=Vec3(0, 0, -1),
                                    up=Vec3(0, -1, 0),
                                    move=Vec3(0, 0, 1),
                                    distance=0.3)

    
    
    with VideoWriter(f"output/mlp_rendering_{PERSON}_{IMAGE_CONFIGURATION}_AllLoss_DEnseFlame{n_iterations}.mp4", (H, W), fps=fps) as video_writer:
        for pose in tqdm(trajectory, desc="Generating video"):
            gs_camera = pose_to_rendercam(pose, intrinsics, W, H)
            output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color)
            rendered_image = output['render']
            rendered_image = Img.from_torch(rendered_image).to_numpy().img
            video_writer.add_image(rendered_image)
    
    print(f" video saved under: output/mlp_rendering_{PERSON}_{IMAGE_CONFIGURATION}_pixelnerf_allLoss{n_iterations}.mp4")
    print('saving gaussian_model as .ply')
    gaussian_model.save_ply(f"FLAME_output/{PERSON}newGS_Flame_PC_{iteration}.ply")
