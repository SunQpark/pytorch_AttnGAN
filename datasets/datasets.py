import sys, os
import numpy as np
import pickle as pkl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import unicodedata
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CubDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        
        self.image_dir = os.path.join(data_dir, 'CUB_200_2011/images')
        self.text_dir = os.path.join(data_dir, 'text')
        
        self.mode = 'train' if train else 'test'
        fname_path = os.path.join(data_dir, f'{self.mode}/filenames.pickle')
        with open(fname_path, 'rb') as fname_file:
            self.fnames = pkl.load(fname_file)
        self.transform = transform
        # self.target_transform = target_transform
        self.EOS_token = 1
        self.preprocessed = self.prepare_dict()
        print('vocabsize', self.preprocessed.n_words)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.image_dir, f'{fname}.jpg')
        text_path = os.path.join(self.text_dir, f'{fname}.txt')

        # open image file, convert to have 3 channels
        data = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            data = self.transform(data)

        # select one sentence from given set of captions
        # with open(text_path, 'r') as text_file:
        #     captions = list(text_file)
        captions = open(text_path, encoding='utf-8').read().strip().replace('.','').split('\n')
        # print('caption', captions)
        select_idx = np.random.randint(len(captions), size=None)
        label_pre = captions[select_idx]

        s = self._unicodeToAscii(label_pre.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        label_pre = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        label = self.tensorFromSentence(self.preprocessed, label_pre)
        return data, label
    
    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def __len__(self):
        return len(self.fnames)

    def prepare_dict(self):
        text_sent = preprocess_text()
        for fname in self.fnames:
            text_path = os.path.join(self.text_dir, f'{fname}.txt')
            captions = open(text_path, encoding='utf-8').read().strip().replace('.','').split('\n')
            s = [text_sent.normalizeString(s) for s in captions]
            for k in range(len(s)):
                text_sent.addSentence(s[k])
        return text_sent

    def _indexesFromSentence(self, pre, sentence):
        return [pre.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, pre, sentence):
        indexes = self._indexesFromSentence(pre, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


class preprocess_text:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self._addWord(word)

    def _addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self._unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        # print('s', len(s))
        return s


class CocoWrapper(datasets.CocoCaptions):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "images/train2014")
        self.ann_dir = os.path.join(self.data_dir, "annotations/captions_train2014.json")
        super(CocoWrapper, self).__init__(self.img_dir, self.ann_dir, transform=transform, target_transform=target_transform)

    def __getitem__(self, idx):
        data, captions = super(CocoWrapper, self).__getitem__(idx)
        captions = list(captions)
        
        select_idx = np.random.randint(len(captions), size=None)
        label_pre = captions[select_idx].replace('\n', '').replace('.','')

        s = label_pre.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        label_pre = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        label = self.tensorFromSentence(self.preprocessed, label_pre)

        return data, label


if __name__ == '__main__':
    trsfm = transforms.Compose([
        transforms.CenterCrop(256),
        # transforms.Resize(64),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                       std=[0.229, 0.224, 0.225])
    ])
    cub_dataset = CubDataset('../data/birds', transform=trsfm)

    # dataloader = DataLoader(cub_dataset, batch_size=24, shuffle=True)
    index = 10
    img, target = cub_dataset[index]
    # for batch, data in enumerate(dataloader):
    #     img = data
    #     print(batch, img.shape)
    print(img[0].shape)
    print(img[1].shape)
    print(img[2].shape)
    print(target)
