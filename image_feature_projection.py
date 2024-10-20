import numpy as np
from gaussian_splatting.utils.sh_utils import SH2RGB
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType, Intrinsics
import math

class ImageFeatureProjection:
    
    
    def project_to_image_coordinates(means, cam_2_world_poses, intrinsics, image_height, image_width):
        means = np.array(means)
        means = np.concatenate((means, np.ones([means.shape[0], 1])), axis=1)
        projected_coordinates_per_image = {}
        distances_per_image = {}
        for image_id, cam_2_world_pose in cam_2_world_poses.items():
            world_2_cam = cam_2_world_pose.change_pose_type(PoseType.WORLD_2_CAM)
            p_cams = world_2_cam @ means.T
            ds = p_cams[2,:]
            p_screens = p_cams / ds
            p_screens = p_screens[0:3,:]
            p_screens = intrinsics @ p_screens
            p_screens = p_screens.T
            p_screens[1,:] += image_height # Axis is at bottom and +y goes further down (OPENCV)
            projected_coordinates_per_image[image_id] = p_screens.astype(int)
            
            distances_per_image[image_id] = ds.T
        return projected_coordinates_per_image, distances_per_image
    
    def project_features_onto_gaussians(xyz, features_per_image, cam_2_world_poses, intrinsics, original_image_height, original_image_width, max_dist_factor=0.5):
        farl_height, farl_width, _ = list(features_per_image.values())[0].shape
        intrinsics = ImageFeatureProjection.rescale_intrinsics_to_farl_dimensions(
            intrinsics, 
            features_per_image, 
            original_image_height, 
            original_image_width
        )
        
        projected_coordinates_per_image, distances_per_image = ImageFeatureProjection.project_to_image_coordinates(
            means=xyz, 
            cam_2_world_poses=cam_2_world_poses, 
            intrinsics=intrinsics,
            image_height=original_image_height,
            image_width=original_image_width
        )
        
        gaussian_features = np.zeros([xyz.shape[0], list(features_per_image.values())[0].shape[2]])
        current_distance = np.ones([xyz.shape[0]]) * math.inf
        for image_id, features in features_per_image.items():
            projected_coordinates = projected_coordinates_per_image[image_id]
            distances = distances_per_image[image_id]
            
            max_dist = distances.min()+(distances.max()-distances.min())*max_dist_factor
            projected_coordinates[distances>max_dist, 1] = 99999
            projected_coordinates[distances>max_dist, 2] = 99999
            
            # Clean up (in view)
            cleaned_ids = ImageFeatureProjection.filter_to_coordinate_ids_in_view(projected_coordinates, farl_height, farl_width)
            cleaned_projected_coordinates = projected_coordinates[cleaned_ids]
            cleaned_distances = distances[cleaned_ids]
            
            # Check if distance is smaller
            distance_smaller = cleaned_distances < current_distance[cleaned_ids]
            cleaned_projected_coordinates = cleaned_projected_coordinates[distance_smaller]
            gaussian_features[cleaned_ids[distance_smaller]] = features[cleaned_projected_coordinates[:,1],cleaned_projected_coordinates[:,0], :]
            current_distance[cleaned_ids[distance_smaller]] = cleaned_distances[distance_smaller]
            
        return gaussian_features
    
    def project_gaussians_onto_images(xyz, cam_2_world_poses, intrinsics, image_height, image_width, gaussian_features=None):
        if gaussian_features is None:
            gaussian_features = np.ones([xyz.shape[0], 1])
        
        projected_coordinates_per_image, distances_per_image = ImageFeatureProjection.project_to_image_coordinates(
            means=xyz, 
            cam_2_world_poses=cam_2_world_poses, 
            intrinsics=intrinsics,
            image_height=image_height,
            image_width=image_width
        )
        
        images = {}
        for image_id, projected_coordinates in projected_coordinates_per_image.items():
            distances = distances_per_image[image_id]
            
            # Clean up (in view)
            cleaned_ids = ImageFeatureProjection.filter_to_coordinate_ids_in_view(projected_coordinates, image_height, image_width)
            cleaned_projected_coordinates = projected_coordinates[cleaned_ids]
            cleaned_distances = distances[cleaned_ids]
            cleaned_gaussian_features = gaussian_features[cleaned_ids]
            
            # Get nearest gaussian per pixel
            #unique_gaussian_ids = ImageFeatureProjection.unique_gaussian_ids_by_distance(cleaned_projected_coordinates, cleaned_distances)
            unique_projected_coordinates = cleaned_projected_coordinates#[unique_gaussian_ids]
            unique_gaussian_features = cleaned_gaussian_features#[unique_gaussian_ids]
            
            image = np.zeros([image_height, image_width, unique_gaussian_features.shape[1]], dtype=int)
            image[unique_projected_coordinates[:, 1], unique_projected_coordinates[:, 0], :] = unique_gaussian_features
            images[image_id] = image
            
        return images
            
    def filter_to_coordinate_ids_in_view(coordinates, image_height, image_width):
        """
        returns the position not the array itself
        """
        width_limited_smaller_width = np.argwhere(coordinates[:,0] < image_width)
        width_limited_bigger_zero = np.argwhere(coordinates[:,0] >= 0)
        height_limited_smaller_height = np.argwhere(coordinates[:,1] < image_height)
        height_limited_bigger_zero = np.argwhere(coordinates[:,1] >= 0)
        
        cleaned_ids = np.intersect1d(np.intersect1d(width_limited_smaller_width, width_limited_bigger_zero), np.intersect1d(height_limited_smaller_height, height_limited_bigger_zero))
        return cleaned_ids    
            
    def unique_gaussian_ids_by_distance(projected_coordinates, distances):
        """
        Ensures that each pixel is only accessed once
        """
        args_sorted_by_distance = np.argsort(distances)
        projected_coordinates_sorted_by_dist = projected_coordinates[args_sorted_by_distance]
        args_unique = np.unique(projected_coordinates_sorted_by_dist, return_index=True)[0]
        
        return args_sorted_by_distance[args_unique]
    
    def rescale_intrinsics_to_farl_dimensions(intrinsics, features_per_image, original_image_height, original_image_width):
        farl_height, farl_width, _ = list(features_per_image.values())[0].shape
        
        return intrinsics.rescale(
            scale_factor=float(farl_width)/float(original_image_width),
            scale_factor_y=float(farl_height)/float(original_image_height),
            inplace=False
        )