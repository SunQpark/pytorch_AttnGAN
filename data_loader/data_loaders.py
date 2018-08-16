import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
sys.path.append('../')
from base import BaseDataLoader
from datasets import *


def resize_images(image, sizes, to_tensor=True):
    if to_tensor:
        resizers = [transforms.Compose([transforms.Resize(s), transforms.ToTensor()]) for s in sizes]
    else:
        resizers = [transforms.Resize(s) for s in sizes]

    images = [r(image) for r in resizers]
    return images


def collate_text(list_inputs):
    order = np.argsort([t.shape[0] for _, t in list_inputs])

    images_sorted = [list_inputs[i][0] for i in order[::-1]]
    text_sorted = [list_inputs[i][1] for i in order[::-1]]
    data = []
    for images in zip(*images_sorted):
        data.append(torch.cat([img.unsqueeze(0) for img in images]))
    target = pack_sequence(text_sorted)
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
            lambda x: resize_images(x, [64, 128, 256])
        ])
        
        self.dataset = CocoWrapper(data_dir, transform=trsfm)
        super(CocoDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=collate_text)

    

class CubDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=0):
        
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
            
        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            lambda x: resize_images(x, [64, 128, 256])
        ])
        
        self.dataset = CubDataset(data_dir, transform=trsfm)
        super(CubDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=collate_text)


if __name__ == '__main__':

    cub_loader = CubDataLoader('../../data/birds', 2)
    
    for i, (data, target) in enumerate(cub_loader):
        # print(data)
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(target)
    #     # padded = pad_packed_sequence(target)
        break
