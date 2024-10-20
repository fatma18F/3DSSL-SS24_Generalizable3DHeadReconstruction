#!/bin/bash
# python preprocess.py 'configs/settings_final_no_encoder.yaml'
# python train.py 'configs/settings_final_no_encoder.yaml'

# python preprocess.py 'configs/settings_final.yaml'
# python train.py 'configs/settings_final.yaml'

# python preprocess.py 'configs/settings_final_no_depth.yaml'
# python train.py 'configs/settings_final_no_depth.yaml'

# python preprocess.py 'configs/settings_final_resnet.yaml'
# python train.py 'configs/settings_final_resnet.yaml'

#python preprocess.py 'configs/settings_final_single_image.yaml'
#python train.py 'configs/settings_final_single_image.yaml'

#python preprocess.py 'configs/settings_final_short.yaml'
#python train.py 'configs/settings_final_short.yaml'

#python train.py 'configs/settings_final_cont.yaml'

python preprocess.py 'configs/settings_final_template_flame.yaml'
python train.py 'configs/settings_final_template_flame.yaml'

#python preprocess.py 'configs/settings_final_lower_lr.yaml'
#python train.py 'configs/settings_final_lower_lr.yaml'
