import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchvision.models import inception_v3
sys.path.append('./')
from base import BaseModel
from torch import Tensor
from torch.nn import Parameter
# from data_loader import CocoDataLoader, CubDataLoader

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.upsample(self.relu(self.bn(self.conv(x))), scale_factor=2)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize='batch'):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        if normalize == 'batch':
            self.norm = nn.BatchNorm2d(out_ch)
        elif normalize == 'inst':
            self.norm = nn.InstanceNorm2d(out_ch)
        elif normalize == 'spectral':
            self.conv = SpectralNorm(self.conv)
            self.norm = None
        else:
            self.norm = None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x_input):
        x = self.conv(x_input)
        if self.norm is not None:
            x = self.norm(x)
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
        # print('text_encoder', output.size())
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
            UpsampleBlock( 8 * n_g,  2 * n_g),
        )
        self.G_0 = nn.Conv2d(2 * n_g, 3, 3, 1, 1)
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
        self.res1 = ResidualBlock(4 * n_g, 4 * n_g)
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
        # print('h_Size', h.size(), 'e_size', e.size())
        s = torch.bmm(e.transpose(1, 2), h)
        beta = F.softmax(s, dim=1) # (b, l, n), attention along word embeddings
        
        # e:(b, f, 1, l) * beta:(b, 1, n, l) 
        c = torch.sum(e.unsqueeze(2) * beta.transpose(1, 2).unsqueeze(1), dim=3)
        return c.view(batch, ch, width, height)

class EB_Discriminator(nn.Module):
    def __init__(self, in_ch, num_downsample=4, embed_size=128, n_d=64):
        super(EB_Discriminator, self).__init__()
        self.downsamples = nn.Sequential(*[DownsampleBlock(in_ch*2**i, in_ch*2**(i+1), 'spectral') for i in range(num_downsample)])
        self.conv = nn.Conv2d(in_ch*2**num_downsample, 8*n_d, 3, 1, 1)
        self.fc_cond = nn.Linear(embed_size, n_d)
        self.conv_cond = nn.Conv2d(9*n_d, 1, 1, 1, 0)
        
        self.embedding = nn.Linear(down_dim, 32)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )
        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, opt.channels, 3, 1, 1)
        )


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

        embedding = self.embedding(score_total.view(score_total.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        
        return score_total


class Discriminator(nn.Module):
    def __init__(self, in_ch, num_downsample=4, embed_size=128, n_d=64, norm_mode='inst'):
        super(Discriminator, self).__init__()
        self.downsamples = nn.Sequential(*[DownsampleBlock(in_ch*2**i, in_ch*2**(i+1), norm_mode) for i in range(num_downsample)])
        if norm_mode == 'spectral':
            self.conv = SpectralNorm(nn.Conv2d(in_ch*2**num_downsample, 8*n_d, 3, 1, 1))
        else:
        self.conv = nn.Conv2d(in_ch*2**num_downsample, 8*n_d, 3, 1, 1)
        
        self.conv_uncond = nn.Conv2d(8*n_d, 1, 1, 1, 0)
        self.fc_cond = nn.Linear(embed_size, n_d)
        self.conv_cond = nn.Conv2d(9*n_d, 1, 1, 1, 0)

    def forward(self, x_input, condition):
        x = self.downsamples(x_input)
        x = F.relu(self.conv(x), inplace=True)
        score_uncond = F.sigmoid(self.conv_uncond(x))
        
        c = self.fc_cond(condition)
        c = c.unsqueeze(2).unsqueeze(2)
        c = c.expand(c.shape[0], c.shape[1], x.shape[2], x.shape[3])
        concat = torch.cat([x, c], dim=1)
        score_cond = F.sigmoid(self.conv_cond(concat))
        score_total = (score_cond + score_uncond) / 2
        # score_total = torch.cat([score_cond, score_uncond], dim=1)
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


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    
    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Generator_module_1(nn.Module):
    def __init__(self, embedding_size, latent_size,vocab_size, hidden_size, num_layer, dropout):
        super(Generator_module_1, self).__init__()
        self.F_ca = F_ca(embedding_size, latent_size)
        self.F_0 = F_0(latent_size)
        self.Text_encoder = Text_encoder(vocab_size, embedding_size, hidden_size, num_layer, dropout)
        self.F_1 = F_1(latent_size)
        self.F_attn = F_attn(embedding_size, hidden_size)
        
    def forward(self, label):
        text_embedded = self.Text_encoder(label)
        sen_feature = torch.mean(text_embedded, dim = 1)
        mu, std, cond = self.F_ca(sen_feature)
        random_noise = torch.randn_like(cond)
        input = torch.cat((random_noise, cond), dim=1)
        h_0, generated_0 = self.F_0(input)
        c_0 = self.F_attn(text_embedded, h_0)
        h_1, generated_1 = self.F_1(c_0, h_0)
        c_1 = self.F_attn(text_embedded, h_1)
        _, generated_2 = self.F_1(c_1, h_1)
        return generated_0, generated_1, generated_2, cond, mu, std

class AttnGAN(nn.Module):
    def __init__(self, embedding_size, latent_size, in_ch, num_downsample, n_d, vocab_size, hidden_size, num_layer, dropout):
        super(AttnGAN, self).__init__()
        self.G = Generator_module_1(embedding_size, latent_size, vocab_size, hidden_size, num_layer, dropout)
        self.D = Discriminator(in_ch, 4)


    def forward(self, label):
        # text_embedded = self.text_encoder(label)
        # generated, cond, mu, std, h_0 = self.G(label)
        # c_0 = self.F_attn(text_embedded, h_0)
        # h_1, x_1 = self.F_1(c_0, h_0)
        # score_1= self.D(x_1, cond)
        # c_1 = self.F_attn(text_embedded, h_1)
        # h_2, x_2 = self.F_1(c_1, h_1)
        # score_2 = self.D(x_2, cond)
        # return score_1, score_2
        pass
        
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Discriminator(3)

    dummy = torch.randn((2, 3, 64, 64))
    dummy_cond = torch.randn((2, 128))
    output = model(dummy, dummy_cond)

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

