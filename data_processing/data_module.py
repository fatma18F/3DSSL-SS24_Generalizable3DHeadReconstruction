from pytorch_lightning import LightningDataModule
from data_processing.dataset import FaceDataset
import torch

def custom_follate_fn(data):
    assert len(data)==1
    return data[0]

class FaceDataModule(LightningDataModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def prepare_data(self):
        pass
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train = FaceDataset(self.config, 'train')
            self.val = FaceDataset(self.config, 'val')

        if stage == 'test' or stage is None:
            self.test = FaceDataset(self.config, 'test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=1, num_workers=self.config['data']['dataloader_num_workers'], collate_fn=custom_follate_fn, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=1, num_workers=self.config['data']['dataloader_num_workers'], collate_fn=custom_follate_fn, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=1, num_workers=self.config['data']['dataloader_num_workers'], collate_fn=custom_follate_fn, shuffle=False)
    
    
        