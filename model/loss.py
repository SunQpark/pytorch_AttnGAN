import torch
# import torch.nn as nn
import torch.nn.functional as F
eps = 1e-8


def gan_loss(output, label):
    # loss = 
    target = torch.full_like(output, label)
    return F.mse_loss(output, target)


def D_loss(out_fake, out_real):
    d_loss = -torch.mean(torch.log(out_real + eps) + torch.log(1 - out_fake + eps))
    return d_loss

def G_loss(out_fake):
    g_loss = -torch.mean(torch.log(out_fake + eps))
    return g_loss
