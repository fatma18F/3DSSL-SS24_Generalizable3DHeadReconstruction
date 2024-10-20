#pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
from pytorch3d.io import load_obj
#pip install chumpy numpy plyfile scipy tqdm 
import numpy as np
import pytorch3d.structures
import torch
import trimesh
from plyfile import PlyData
import flame.flame as imfp
import flame.lbs as lbs
from dreifus.matrix import Pose, Intrinsics
import trimesh
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes

class FlameModelCreation():        
    @classmethod
    def create_flame_model(cls, tracked_flame_params_path, config):
        use_template_flame = config['data'].get('use_template_flame', False)
        flame_model = imfp.FlameHead(shape_params=300, expr_params=100)
        if use_template_flame:
            print('using template flame')
            flame_vertices = flame_model.v_template.clone().unsqueeze(0)
        else:
            print('using presonalized flame')
            flame_data = np.load(tracked_flame_params_path)
            
            shape_params = torch.tensor(np.tile(flame_data['shape'], (128, 1)))
            expression_params = torch.tensor(flame_data['expression'])
            pose_params=torch.tensor(FlameModelCreation._create_pose_param(flame_data['jaw']))
            neck_pose = torch.tensor(flame_data['neck'])
            eye_pose = torch.tensor(flame_data['eyes'])
            jaw = torch.tensor(flame_data['jaw'])
            
            # FLAME MODEL
            i=0 #timestep
            flame_vertices, flame_lms = flame_model.forward3(
                shape=shape_params[[i]],  # We always assume the same shape params for all timesteps
                expr=expression_params[[i]],
                rotation=None,#rotation[[i]],
                neck=neck_pose[[i]], 
                jaw=jaw[[i]],           
                eyes= eye_pose[[i]],
                translation=None,
                pose_params=pose_params[[i]]
            )
        
        # model transformation
        flame_vertices = FlameModelCreation._upsample_flame(flame_vertices, flame_model.faces, config)
        
        return flame_vertices.squeeze().cpu().numpy()
    
    @classmethod
    def save_flame_model(cls, vertices, flame_path):
        with open(flame_path, 'w') as file:
            for vertex in vertices:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")    
                
    @classmethod
    def get_transformation_and_scale(cls, tracked_flame_params_path):
        timestep = 0
        flame_data = np.load(tracked_flame_params_path)
        transl = torch.tensor(flame_data['translation'])
        rotation = torch.tensor(flame_data['rotation'])
        scale = torch.tensor(flame_data['scale'])
        
        model_transformations = torch.stack([torch.from_numpy(
                        Pose.from_euler(rotation[timestep].numpy(), transl[timestep].numpy(), 'XYZ'))])
        model_transformations[:, :3, :3] *= scale[timestep]
        
        return model_transformations.squeeze(0)
    
    @classmethod
    def _upsample_flame_alt(cls, flame_vertices, flame_faces, config):
        # Probably worse as points might not have the good structure from flame
        # And relative positions wont be the same between faces
        number_of_points_sampled_from_flame = config['data']['number_of_points_sampled_from_flame']
        if len(flame_faces.shape) == 2:
            flame_faces = flame_faces.unsqueeze(0)
        flame_vertices = flame_vertices.float()
        #flame_faces = flame_faces.float()
        p3d_mesh = pytorch3d.structures.Meshes(flame_vertices, flame_faces)
        points = sample_points_from_meshes(p3d_mesh, number_of_points_sampled_from_flame, return_normals=False)
        return points
    
    @classmethod
    def _upsample_flame(cls, flame_vertices, flame_faces, config):
        if not config['data']['upsample_flame_iterations']:
            return flame_vertices
        upsample_flame_iterations = config['data']['upsample_flame_iterations']
        if len(flame_faces.shape) == 2:
            flame_faces = flame_faces.unsqueeze(0)
        p3d_mesh = pytorch3d.structures.Meshes(flame_vertices, flame_faces)
        p3d_mesh_subdivision = pytorch3d.ops.SubdivideMeshes()
        
        for _ in range(upsample_flame_iterations):
            p3d_mesh = p3d_mesh_subdivision(p3d_mesh)
            
        vertices = p3d_mesh.verts_list()[0]
        faces = p3d_mesh.faces_list()[0]
        #selected_vertex_indices = [i for i in range(vertices.shape[0]) if i%6!=0]
        #vertices = vertices[selected_vertex_indices, :]
        return vertices.unsqueeze(0)
    
    @classmethod    
    def _create_pose_param(cls, jaw_pose):
        #jaw_pose = flame_data['jaw']  # Shape: (128, 3)
        num_samples = jaw_pose.shape[0]

        pose_params = np.zeros((num_samples, 6))  # 6 parameters: [0, 0, 0, jaw_x, jaw_y, jaw_z]
        pose_params[:, 3:] = jaw_pose
        return pose_params