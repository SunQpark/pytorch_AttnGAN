import sys
sys.path.append('./')
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from data_loader import CocoDataLoader, CubDataLoader


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.upsample(self.relu(self.bn(self.conv(x))), scale_factor=2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x_input):
        residual = x_input
        x = self.relu(self.bn1(self.conv1(x_input)))
        x = self.bn2(self.conv2(x))
        return x + residual


class Text_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, dropout):
        super(Text_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=None)
        self.bi_lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layer, dropout=dropout, bidirectional=True)

    def forward(self, x):
        x, lengths = pad_packed_sequence(x)
        x = x.squeeze()
        embedded = self.embedding(x.cpu())
        embedded = pack_padded_sequence(embedded, lengths)
        output, _ = self.bi_lstm(embedded)
        output, _ = pad_packed_sequence(output)
        # encoded = torch.cat((output[:, :,:self.hidden_size], output[:, :,self.hidden_size :]),2)
        return output


class F_ca(nn.Module):
    """ conditioning augmentation of sentence embedding """
    def __init__(self, embedding_size, latent_size):
        super(F_ca, self).__init__()
        self.fc_mu = nn.Linear(embedding_size, latent_size)
        self.fc_std = nn.Linear(embedding_size, latent_size)

    def forward(self, e_input):
        mu = self.fc_mu(e_input)
        if self.training:
            std = self.fc_std(e_input)
            eps = torch.randn_like(fc_std)
            return mu + std * eps
        else:
            return mu


class F_0(nn.Module):
    def __init__(self, latent_size, n_g=32):
        super(F_0, self).__init__()
        self.fc = nn.Linear(latent_size, 4 * 4 * 64 * n_g)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64 * n_g, 32 * n_g),
            UpsampleBlock(32 * n_g, 16 * n_g),
            UpsampleBlock(16 * n_g,  8 * n_g),
            UpsampleBlock( 8 * n_g,  4 * n_g),
        )
        self.G_0 = nn.Conv2d(4 * n_g, 3, 3, 1, 1)

    def forward(self, z):
        z = self.fc(z).view(-1, 64*self.n_g, 4, 4)
        h_0 = self.upsample_blocks(z)

        # generator block
        output = (F.tanh(self.G_0(h_0)) + 1) / 2
        return h_0, output


class F_1(nn.Module):
    def __init__(self, in_ch, n_g=32):
        super(F_1, self).__init__()
        self.res1 = ResidualBlock(in_ch,   4 * n_g)
        self.res2 = ResidualBlock(4 * n_g, 4 * n_g)
        self.upsample = UpsampleBlock(4 * n_g, 8 * n_g)

    def forward(self, h_0, c):
        pass


class F_attn(BaseModel):
    def __init__(self, batch_size, h_dim, e_dim, h_dim_2, e_dim_2):
        super(F_attn, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(h_dim, e_dim)
        self.h_dim_2 = h_dim_2
        self.e_dim_2 = e_dim_2
        self.h_dim = h_dim
        self.batch_size = batch_size

    def forward(self, e, h):
        e_p = self.linear(e)
        e_p = torch.transpose(e_p,1,2)
        h = torch.transpose(h,1,2)
        s = torch.bmm(h, e_p)
        beta = self.softmax(s)
        c = torch.zeros(self.batch_size, self.h_dim, self.h_dim_2)
        for n in range(self.h_dim_2):
            for k in range(self.e_dim_2):
                c[:,:, n] = torch.mul(e_p[:,:, k], beta[:,n, k])
        return c


class AttnGAN(BaseModel):
    def __init__(self, parameter_list):
        super(AttnGAN, self).__init__()

    def forward(self, parameter_list):
        pass


if __name__ == '__main__':
    #test F_attn
    model = F_attn(10, 10, 5, 5)
    model.cuda()
    e = torch.randn((10,5,10), device='cuda')
    h = torch.randn((10,10,5), device='cuda')
    output = model(e,h)
    print(output.shape)

    #Test text_encoder
#
#
# if __name__ == '__main__':
#     data_loader = CubDataLoader('../../data/birds', 4)
#     model = Text_encoder(4800, 100, 100, 2, 0.3)
#     for batch_idx, (data, target) in enumerate(data_loader):
#         output = model(target)
#         print(output.shape, output)
#         break
#     # print(x.shape)
#     # print(out.shape)
