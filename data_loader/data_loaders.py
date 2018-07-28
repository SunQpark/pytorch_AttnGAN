import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
sys.path.append('./')
from base import BaseDataLoader
from datasets import *


def collate_text(list_inputs):
    order = np.argsort([t.shape[0] for d, t in list_inputs])
    list_sorted = [list_inputs[i][1] for i in order[::-1]]
    data = torch.cat([list_inputs[i][0].unsqueeze(0) for i in order[:-1]])
    target = pack_sequence(list_sorted)
    return data, target


class SVHNDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = datasets.SVHN(data_dir, split='train', transform=trsfm, download=True)
        super(SVHNDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers)


class CocoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
            
        self.img_dir = os.path.join(data_dir, "images/train2014")
        self.ann_dir = os.path.join(data_dir, "annotations/captions_train2014.json")

        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(64),
            transforms.ToTensor(),
        ])
        
        self.dataset = CocoWrapper(data_dir, transform=trsfm)
        super(CocoDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=collate_text)

    

class CubDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=0):
        
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
            
        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Resize(64),
            transforms.ToTensor(),
        ])
        
        self.dataset = CubDataset(data_dir, transform=trsfm)
        super(CubDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=collate_text)


if __name__ == '__main__':
    # coco_loader = CocoDataLoader('../cocoapi', 4)
    cub_loader = CubDataLoader('../data/birds', 4)
    
    for i, (data, target) in enumerate(cub_loader):
        print(data.shape)
        print(i, target)
        # padded = pad_packed_sequence(target)
        break
