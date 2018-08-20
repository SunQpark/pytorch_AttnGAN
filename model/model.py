import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchvision.models import inception_v3
sys.path.append('./')
# from base import BaseModel
# from data_loader import CocoDataLoader, CubDataLoader

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        x = self.relu(self.bn(self.conv(x)))
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, inst_norm=True):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        if inst_norm:
            self.inorm = nn.BatchNorm2d(out_ch)
        else:
            self.inorm = None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x_input):
        x = self.conv(x_input)
        if self.inorm is not None:
            x = self.inorm(x)
        x = self.lrelu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, ch, stride=1):
        super(ResidualBlock, self).__init__()
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
        embedded = self.embedding(x)
        embedded = pack_padded_sequence(embedded, lengths)
        output, _ = self.bi_lstm(embedded)
        output, _ = pad_packed_sequence(output)
        output = output.transpose(0,1)
        return output


class Image_encoder(nn.Module):
    def __init__(self):
        super(Image_encoder, self).__init__()
        inception = inception_v3(pretrained=True)
        inception.eval()
        self.layers1 = nn.Sequential(*list(inception.children())[:-5]) # to mixed_6e
        self.layers2 = nn.Sequential(*list(inception.children())[-4:-1])

    def forward(self, x):
        x = self.layers1(x) # input size: (80, 80)
        output = self.layers2(x)
        output = F.adaptive_avg_pool2d(output, 1)
        return x, output


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
            eps = torch.randn_like(std)
            return mu, std, mu + std * eps
        else:
            return mu


class F_0(nn.Module):
    def __init__(self, latent_size, n_g=32):                                         
        super(F_0, self).__init__()
        self.fc = nn.Linear(latent_size*2, 4 * 4 * 64 * n_g)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64 * n_g, 32 * n_g),
            UpsampleBlock(32 * n_g, 16 * n_g),
            UpsampleBlock(16 * n_g,  8 * n_g),
            UpsampleBlock( 8 * n_g,  4 * n_g),
        )
        self.G_0 = nn.Conv2d(4 * n_g, 3, 3, 1, 1)
        self.n_g = n_g

    def forward(self, z):
        z = self.fc(z).view(-1, 64*self.n_g, 4, 4)
        h_0 = self.upsample_blocks(z)

        # generator block
        output = (F.tanh(self.G_0(h_0)) + 1) / 2
        return h_0, output


class F_1(nn.Module):
    def __init__(self, in_ch, n_g=32):
        super(F_1, self).__init__()
        self.res1 = ResidualBlock(8 * n_g, 4 * n_g)
        self.res2 = ResidualBlock(4 * n_g, 4 * n_g)
        self.upsample = UpsampleBlock(4 * n_g, 2 * n_g)

        self.G_1 = nn.Conv2d(2 * n_g, 3, 3, 1, 1)

    def forward(self, h_0, c):
        joined = torch.cat([h_0, c], dim=1)
        h = self.res1(joined)
        h = self.res2(h)
        h = self.upsample(h)
        
        x = (F.tanh(self.G_1(h)) + 1) / 2
        return h, x


class F_attn(nn.Module):
    def __init__(self, e_dim, h_dim):
        super(F_attn, self).__init__()
        self.linear = nn.Linear(e_dim, h_dim)

    def forward(self, e, h):
        e = self.linear(e) # (b, l, f)
        e = e.transpose(1, 2) # (b, f, l)

        batch, ch, width, height = h.shape
        h = h.view(batch, ch, -1) # (b, f, n), flatten img features by subregions, n = wid * hei
        s = torch.bmm(e.transpose(1, 2), h)
        beta = F.softmax(s, dim=1) # (b, l, n), attention along word embeddings
        
        # e:(b, f, 1, l) * beta:(b, 1, n, l) 
        c = torch.sum(e.unsqueeze(2) * beta.transpose(1, 2).unsqueeze(1), dim=3)
        return c.view(batch, ch, width, height)


class Discriminator(nn.Module):
    def __init__(self, in_ch, num_downsample=4, embed_size=100, n_d=64):
        super(Discriminator, self).__init__()
        self.downsamples = nn.Sequential(*[DownsampleBlock(in_ch*2**i, in_ch*2**(i+1)) for i in range(num_downsample)])
        self.conv = nn.Conv2d(in_ch*2**num_downsample, 8*n_d, 3, 1, 1)
        self.fc_cond = nn.Linear(embed_size, n_d)
        self.conv_cond = nn.Conv2d(9*n_d, 1, 1, 1, 0)

    def forward(self, x_input, condition):
        x = self.downsamples(x_input)
        x = self.conv(x)
        score_uncond = F.sigmoid(x)
        
        c = self.fc_cond(condition)
        c = c.unsqueeze(2).unsqueeze(2)
        c = c.expand(c.shape[0], c.shape[1], x.shape[2], x.shape[3])

        concat = torch.cat([x, c], dim=1)
        score_cond = F.sigmoid(self.conv_cond(concat))
        score_total = torch.cat([score_cond, score_uncond], dim=1)
        return score_total

class Matching_Score_word(nn.Module):
    def __init__(self, gamma_1, gamma_2, gamma_3):
        super(Matching_Score_word, self).__init__()
        self.cos = nn.CosineSimilarity(dim = 1)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        
    def batch_score(self, e, v):
        s = torch.mm(e.transpose(0,1), v) #(T, 289)
        s_norm = F.softmax(s, dim=1) #( T, 289) 
        alpha = F.softmax(self.gamma_1 * s_norm, dim=0)# (T, 289)
        
        alpha = torch.unsqueeze(alpha, dim = 1) # (T, 1, 289)
        v= torch.unsqueeze(v, dim=0) # (1, D, 289)
        c = torch.sum(alpha * v, dim=2) # (T, D)
        
        R = self.cos(c.transpose(0,1), e) #(T)
        match_score = torch.log(torch.exp(self.gamma_2 * R).sum(dim=0)).pow(1/self.gamma_2) #scalar
        return match_score

    def forward(self, e, v):
        batch_size = e.shape[0]
        batch_sum_score_d = torch.zeros(batch_size, device = device)
        batch_sum_score_q = torch.zeros(batch_size, device = device)

        for i in range(batch_size):
            for j in range(batch_size):
                batch_sum_score_d[i] += self.batch_score(e[i], v[j])
                batch_sum_score_q[i] += self.batch_score(e[j], v[i])
        return batch_sum_score_d, batch_sum_score_q


class Matching_Score_sent(nn.Module):
    def __init__(self, gamma_3):
        super(Matching_Score_sent, self).__init__()
        self.cos = nn.CosineSimilarity(dim = 1)
        self.gamma_3 = gamma_3

    def forward(self, e_global, v_global):
        batch_size = e_global.shape[0]
        batch_sum_score_d = torch.zeros(batch_size, device = device)
        batch_sum_score_q = torch.zeros(batch_size, device = device)

        for i in range(batch_size):
            for j in range(batch_size):
                batch_sum_score_d[i] += torch.exp(self.gamma_3 *(self.cos(e_global[i], v_global[j])))
                batch_sum_score_q[i] += torch.exp(self.gamma_3 *(self.cos(e_global[j], v_global[i])))
        
        return batch_sum_score_d, batch_sum_score_q

class Generator_module(nn.Module):
    def __init__(self, embedding_size, latent_size,vocab_size, hidden_size, num_layer, dropout):
        super(Generator_module, self).__init__()
        self.F_ca = F_ca(embedding_size, latent_size)
        self.F_0 = F_0(latent_size)
        self.Text_encoder = Text_encoder(vocab_size, embedding_size, hidden_size, num_layer, dropout)

    def forward(self, label):
        sen_feature = torch.mean(self.Text_encoder(label), dim = 1)
        mu, std, cond = self.F_ca(sen_feature)
        random_noise = torch.randn_like(cond)
        input = torch.cat((random_noise, cond), dim=1)
        _, generated = self.F_0(input)
        return generated, cond, mu, std

class AttnGAN(nn.Module):
    def __init__(self, embedding_size, latent_size, in_ch, num_downsample, n_d, vocab_size, hidden_size, num_layer, dropout):
        super(AttnGAN, self).__init__()
        # self.F_ca = F_ca(fca_embedding_size, latent_size)
        # self.F_0 = F_0(latent_size)
        # self.F_1 = F_1(in_ch, n_g)
        # self.F_attn = F_attn(e_dim, h_dim)
        # self.Text_encoder = Text_encoder(vocab_size, word_embedding_size, hidden_size, num_layer, dropout)
        self.G = Generator_module(embedding_size, latent_size, vocab_size, hidden_size, num_layer, dropout)
        self.D = Discriminator(in_ch, num_downsample, latent_size, n_d)

    def forward(self, text):
        generated, cond, mu, std = self.G(text)
        output = self.D(generated, cond)
        return output, mu, std
    
if __name__ == '__main__':
    #Test Image_encoder
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Image_encoder()

    dummy = torch.randn((2, 3, 80, 80))
    output = model(dummy)

    print(output[0].shape)
    print(output[1].shape)
    # #test F_attn
    # model = Matching_Score_word(1,1,1)
    # model= model.to(device)

    # batch_size = 4
    # seq_length = 32
    # seq_features = 20

    # e = torch.randn((batch_size, seq_features,seq_length), device = device)
    # v = torch.randn((batch_size, seq_features, 289), device=device)
    
    # output1, output2 = model(e,v)
    # print(output1.shape, output2.shape)
    # img_channels = 16

    # model = F_attn(seq_features, img_channels)
    # model = model.to(device)
    # print(model)

    # e = torch.randn((batch_size, seq_length, seq_features), device=device)
    # h = torch.randn((batch_size, img_channels, 30, 40), device=device)

    # print('word embedding e: ')
    # print(e.shape)

    # print('hidden features h: ')
    # print(h.shape)

    # output = model(e, h)
    # print('output shape: ')
    # print(output.shape)

    #Test text_encoder
# if __name__ == '__main__':
#     data_loader = CubDataLoader('../../data/birds', 4)
#     model = Text_encoder(4800, 100, 100, 2, 0.3)
#     for batch_idx, (data, target) in enumerate(data_loader):
#         output = model(target)
#         print(output.shape, output)
#         break
#     # print(x.shape)
#     # print(out.shape)

