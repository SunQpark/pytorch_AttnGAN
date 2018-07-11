import sys
sys.path.append('./')
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from data_loader import CocoDataLoader, CubDataLoader

class DC_Generator(BaseModel):

    def __init__(self, n_z, n_c=3, n_size=4):
        '''
        generated image will have shape (batch, n_c, n_size*32, n_size*32) 
        '''
        super(DC_Generator, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(n_z, 1024, n_size, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512,  256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256,  128, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128,  n_c, 4, 2, 1, bias=False)

    def forward(self, z):
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = F.relu(self.bn4(self.conv4(z)))
        return F.tanh(self.conv5(z))

class DC_Discriminator(BaseModel):
    def __init__(self, n_c=3, n_size=4):
        super(DC_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu_(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu_(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu_(self.bn4(self.conv4(x)), 0.2)
        return F.sigmoid(self.conv5(x))


class DC_GAN(BaseModel):
    def __init__(self, n_z, n_c, n_size):
        super(DC_GAN, self).__init__()
        self.G = DC_Generator(n_z, n_c, n_size)
        self.D = DC_Discriminator(n_c, n_size)

    def forward(self, z, x_real):
        x_fake = self.G(z)
        data_pair = torch.cat([x_real, x_fake], dim=0)
        score = self.D(data_pair)
        return score, x_fake

class Text_encoder(BaseModel):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, dropout):
        super(Text_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=None)
        self.bi_lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layer, dropout=dropout, bidirectional=True)

    def forward(self, x):
        x, len = pad_packed_sequence(x)
        x = x.squeeze()
        embedded = self.embedding(x.cpu())
        embedded = pack_padded_sequence(embedded, len)
        output, _ = self.bi_lstm(embedded)
        output, _ = pad_packed_sequence(output)
        # encoded = torch.cat((output[:, :,:self.hidden_size], output[:, :,self.hidden_size :]),2)
        return output

if __name__ == '__main__':
    data_loader = CubDataLoader('../../data/birds', 4)
    model = Text_encoder(4800, 100, 100, 2, 0.3)
    for batch_idx, (data, target) in enumerate(data_loader):
        output = model(target)
        print(output.shape, output)
        break
    # print(x.shape)
    # print(out.shape)
