import json
from argparse import ArgumentParser
from pathlib import Path
import open3d as o3d
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
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.viewer import GaussianViewer
from mediapy import VideoWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
######
#import os
#os.environ['TRAME_DEFAULT_HOST'] = 'mydomain.com'


if __name__ == '__main__':
    use_visualization_window = True
    use_viewer = True
    n_iterations = 10_000#10000
    tracker2='EMO-1-shout+laugh'
    tracker1='SEN-10-port_strong_smokey'
    PERSON='017'
    FLAME_PATH=f'FLAME_output/Flame_{PERSON}.obj'
    TRACKER=tracker2


    downscale_factor = 2.0 #2  # By how much to downscale the images

    # ==========================================================
    # Data
    # ==========================================================

    # Load camera poses and intrinsics
    #camera_params = json.load(open('data/camera_params.json'))
    camera_params = json.load(open(f'/home/ayed/Flame_fittings/Ryoto-single-timestep/{PERSON}/calibration/camaera_params.json'))
    intrinsics = Intrinsics(camera_params['intrinsics'])  # Note: In the demo, all cameras have the same intrinsics
    intrinsics = intrinsics.rescale( 1.0 / (downscale_factor*2))  # Note: when images are rescaled, intrinsics has to be scaled as well!
    cam_2_world_poses = dict()  # serial => world_2_cam_pose
    #for serial, cam_2_world_pose in camera_params['cam_2_world_poses'].items():
    for serial, cam_2_world_pose in camera_params['world_2_cam'].items():
        cam_2_world_pose = Pose(cam_2_world_pose, pose_type=PoseType.WORLD_2_CAM)
        cam_2_world_pose.change_pose_type(PoseType.CAM_2_WORLD)
        cam_2_world_poses[f'cam_{serial}'] = cam_2_world_pose

    # Load images
    tracker2='EMO-1-shout+laugh'
    tracker1='SEN-10-port_strong_smokey'
    images = dict()  # serial => image
    

    serials = []
    for image_file in Path(f'/home/ayed/Flame_fittings/Ryoto-single-timestep/{PERSON}/sequences/{TRACKER}/timesteps/frame_00000/images-2x').iterdir():
        serial = image_file.stem
        image = Image.open(image_file)

        alpha_map_file = str(image_file).replace("/images-2x/", "/alpha_map/").replace("jpg", "png")
        alpha_map = Image.open(alpha_map_file)
        alpha_map = alpha_map.resize((int(image.width / downscale_factor), int(image.height / downscale_factor)))  # Resize image
        alpha_map = np.array(alpha_map)
        mask = ~alpha_map.astype(bool)


        image = image.resize((int(image.width / downscale_factor), int(image.height / downscale_factor)))  # Resize image
        image = np.array(image)
        image[mask] = [255, 255, 255]   
        
        #result_image = Image.fromarray(image)
        ##result_image.save(f'/home/ayed/{serial}.png')
        
        images[serial] = image
        serials.append(f'{serial}')
        #print(serial)

    # ==========================================================
    # Visualize
    # ==========================================================

    pv.start_xvfb()
    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    for serial in serials:
        add_camera_frustum(p, cam_2_world_poses[serial], intrinsics, image=images[serial])
    #p.show()

    # ==========================================================
    # Model
    # ==========================================================

    # Setup 3D Gaussian Model and optimizer
    device = torch.device('cuda')
    gaussian_model = GaussianModel(sh_degree=0)
    bg_color = torch.ones(3, device=device, dtype=torch.float32)  # Use white background for rendering
    rgb_values = [130, 140, 141]
    #bg_color = torch.tensor(rgb_values, dtype=torch.float32, device=device)
    #print('bg_color:',bg_color)

    
    #pointcloud = BasicPointCloud.from_obj(FLAME_PATH)# obj_mesh_50k_point.
    pointcloud = BasicPointCloud(points=torch.randn((5000, 3)) / 20, colors=torch.randn((5000, 3)), normals=None)

    gaussian_model.create_from_pcd(pointcloud, spatial_lr_scale=1)

    # Setup optimizer
    parser = ArgumentParser()
    params_opt = OptimizationParams(parser)
    gaussian_model.training_setup(params_opt)
    optimizer = gaussian_model.optimizer

    # ==========================================================
    # Viewer
    # ==========================================================

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

    image_buffer = None
    if use_visualization_window:
        image_buffer = np.zeros_like(images[serials[0]], dtype=np.float32)
        image_window = ImageWindow(image_buffer)

    # ==========================================================
    # Optimization Loop
    # ==========================================================

    H, W, _ = images[serials[0]].shape
    torch_images = {serial: torch.tensor(image / 255, device=device).permute(2, 0, 1) for serial, image in images.items()}
    progress = tqdm(range(n_iterations))
    for iteration in progress:
        #print(iteration)
        serial = serials[iteration % len(serials)]
        gt_image = torch_images[serial]
        cam_2_world_pose = cam_2_world_poses[serial]
        gs_camera = pose_to_rendercam(cam_2_world_pose, intrinsics, W, H)

        # Render current 3D reconstruction
        output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color,return_depth=False)
        rendered_image = output['render']
        
        # Compute L1 loss
        loss = (rendered_image - gt_image).abs().mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Gaussian Splatting Training tricks:
        #  1. Learning rate schedule for 3D positions (positions will receive smaller and small updates)
        gaussian_model.update_learning_rate(iteration)

        # #  2. Adaptive density control:
        # #       - Clone/Split Gaussians with high positional gradient (that move around a lot)
        # #       - Prune Gaussians with low opacity (don't clutter scene)
        # if (iteration + 1) % 100 == 0:
        #     gaussian_model.add_densification_stats(output['viewspace_points'],
        #                                            output['visibility_filter'])  # Important, otherwise prune and densify don't work
        #     gaussian_model.densify_and_prune(0.0002, 0.005, 1, None)

        #  3. Opacity reset (needed such that Pruning is more effective)
        if (iteration + 1) % 3000 == 0:
            gaussian_model.reset_opacity()

        # Log prediction to visualization window
        if use_visualization_window and serial == 'cam_22200037':
            image_buffer[:] = rendered_image.permute(1, 2, 0).detach().cpu().numpy()

        progress.set_postfix({"loss": loss.item()})
    

    print('saving gaussian_model as .ply')
    gaussian_model.save_ply(f"/home/ayed/Flame_fittings/GaussianAvatars/Point_clouds/{PERSON}newGS_Flame_PC_{iteration}.ply")

        #break 
    #while (True):
    #    pass
    # Generate video
    fps = 24
    seconds = 4
    trajectory = circle_around_axis(seconds * fps,
                                    axis=Vec3(0, 0, -1),
                                    up=Vec3(0, -1, 0),
                                    move=Vec3(0, 0, 1),
                                    distance=0.3)

    with VideoWriter(f"output/rendering_GS_{PERSON}_{n_iterations}_.mp4", (H, W), fps=fps) as video_writer:
        for pose in tqdm(trajectory, desc="Generating video"):
            gs_camera = pose_to_rendercam(pose, intrinsics, W, H)
            output = render(gs_camera, gaussian_model, PipelineParams2(), bg_color)
            rendered_image = output['render']
            rendered_image = Img.from_torch(rendered_image).to_numpy().img
            video_writer.add_image(rendered_image)
