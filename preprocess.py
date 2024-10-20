import yaml
from data_processing.dataset import FaceDataset
from data_processing.data_module import FaceDataModule
from tqdm import tqdm
import FaRL.farl_segmentation_and_features as farl
import argparse

def main(config):
    config['data']['force_rebuild'] = True
    config['data']['dataloader_num_workers'] = 3

    # if config['data']['encoder'] == 'F':
    #     # FaRL features
    #     farl.main(config)
    
    face_data_module = FaceDataModule(config=config)
    face_data_module.setup('fit')
    for data_loader in [face_data_module.train_dataloader(), face_data_module.val_dataloader()]:
        print(f'preprocessing {data_loader.dataset.data_split} split')
        for batch in tqdm(data_loader):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config yaml')
    args = parser.parse_args()
    main(yaml.safe_load(open(args.config, 'r')))