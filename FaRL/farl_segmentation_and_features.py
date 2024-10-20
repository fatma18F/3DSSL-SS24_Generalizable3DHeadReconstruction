from typing import Optional, Dict, Any
import functools
import torch
import torch.nn.functional as F
import facer
import os
from PIL import Image
from facer.util import bchw2hwc
from facer.face_parsing.farl import pretrain_settings
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import yaml

def save_img(image: torch.Tensor, img_path, mode):
    image=bchw2hwc(image)
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())
    pimage.save(img_path.replace('.jpg', f'_{mode}.bmp'))
    
def save_numpy(features, img_path, ending):
    np.save(img_path.replace('.jpg', ending), features.astype(np.float32))
    
def custom_forward(model, images: torch.Tensor, data: Dict[str, Any]):
        setting = pretrain_settings[model.conf_name]
        images = images.float() / 255.0
        _, _, h, w = images.shape

        simages = images[data['image_ids']]
        matrix = setting['get_matrix_fn'](data[setting['matrix_src_tag']])
        grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)
        
        w_backbone_results, _ = model.net.backbone(w_images)
        w_seg_logits = model.net.head(w_backbone_results)  # (b*n) x c x h x w
        
        # backbone_results = F.grid_sample(
        #     w_backbone_results[0], inv_grid, mode='bilinear', align_corners=False)
        # seg_logits = F.grid_sample(
        #     w_seg_logits, inv_grid, mode='bilinear', align_corners=False)

        data['seg'] = {'logits': w_seg_logits,
                       'backbone_results': w_backbone_results[0].detach().cpu(),
                       'label_names': setting['label_names']}
        return data, inv_grid

def read_hwc(path: str, config) -> torch.Tensor:
    """Read an image from a given path.

    Args:
        path (str): The given path.
    """
    downscale_factor = config['data']['downscale_factor']
    image = Image.open(path)
    image = image.resize((int(image.width / downscale_factor), int(image.height / downscale_factor)))
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)

def face_parsing(img_path, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(read_hwc(img_path, config)).to(device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    
    if not faces:
        print(f"ERROR: no faces detected")
        return -1
    if faces['scores'].shape[0] > 1:
        max_prob = faces['scores'].argmax()
        for key, val in faces.items():
            faces[key] = val[max_prob]
            if key in ['rects', 'points', 'image_ids']:
                faces[key] = faces[key].unsqueeze(0)
    
    face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"

    with torch.inference_mode():
        faces, inv_grid = custom_forward(face_parser, image, faces)

    seg_logits = faces['seg']['logits']
    backbone_results = faces['seg']['backbone_results']
    
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    save_img(vis_img.unsqueeze(1), img_path, 'bhw')
    #save_img(facer.draw_bchw(downsized_image, faces), img_path, 'bchw')
    
    concatenated_result = torch.cat([backbone_results, seg_probs.detach().cpu()], dim=1).numpy()
    save_numpy(concatenated_result, img_path, '_farl.npy')
    save_numpy(inv_grid.cpu().numpy(), img_path, '_inv_grid.npy')
  
def main(config):
    dir = config['data']['path']
    failed_images = []
    for person in tqdm(os.listdir(dir)):
        if person in ['_old', 'old_flames']:
            continue
        for pose in os.listdir(f"{dir}/{person}/sequences"):
            for timestep in os.listdir(f"{dir}/{person}/sequences/{pose}/timesteps"):
                for img in os.listdir(f"{dir}/{person}/sequences/{pose}/timesteps/{timestep}/images-2x"):
                    if img.endswith('.jpg'):
                        path = f"{dir}/{person}/sequences/{pose}/timesteps/{timestep}/images-2x/{img}"
                        if False and os.path.exists(path.replace('.jpg', '_farl.npy')):
                            print(f'SKIPPING: {path}')
                        else:
                            print(path)
                            ret_code = face_parsing(path, config)
                            
                            if ret_code==-1:
                                failed_images.append(path)
                                
    print(f'{" FAILED IMAGES":#^40}')
    [print(path) for path in failed_images]
    
if __name__ == '__main__':
    config = yaml.safe_load(open('../configs/settings_latent_codes.yaml', 'r'))
    main()                  
                            