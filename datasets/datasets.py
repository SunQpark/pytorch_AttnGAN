import sys, os
import numpy as np
import pickle as pkl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


class CubDataset(Dataset):

    def __init__(self, data_dir, transform=None, target_transform=None, train=True):
        
        self.image_dir = os.path.join(data_dir, 'CUB_200_2011/images')
        self.text_dir = os.path.join(data_dir, 'text')
        
        self.mode = 'train' if train else 'test'
        fname_path = os.path.join(data_dir, f'{self.mode}/filenames.pickle')
        with open(fname_path, 'rb') as fname_file:
            self.fnames = pkl.load(fname_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.image_dir, f'{fname}.jpg')
        text_path = os.path.join(self.text_dir, f'{fname}.txt')

        # open image file, convert to have 3 channels
        data = Image.open(image_path).convert("RGB")
        
        # select one sentence from given set of captions
        with open(text_path, 'r') as text_file:
            captions = list(text_file)
        select_idx = np.random.randint(len(captions), size=None)
        label = captions[select_idx].replace('\n', '')

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data, label

    def __len__(self):
        return len(self.fnames)


class CocoWrapper(datasets.CocoCaptions):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "images/train2014")
        self.ann_dir = os.path.join(self.data_dir, "annotations/captions_train2014.json")
        super(CocoWrapper, self).__init__(self.img_dir, self.ann_dir, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):
        data, target = super(CocoWrapper, self).__getitem__(idx)
        target = list(target)
        
        select_idx = np.random.randint(len(target), size=None)
        label = target[select_idx].replace('\n', '')
        return data, label


if __name__ == '__main__':
    trsfm = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    cub_dataset = CubDataset('../data/birds', transform=trsfm, target_transform=None)

    dataloader = DataLoader(cub_dataset, batch_size=24, shuffle=True)
    index = 10
    img, target = cub_dataset[index]
    # for batch, data in enumerate(dataloader):
    #     img = data
    #     print(batch, img.shape)
    print(img.shape)
    print(target)
