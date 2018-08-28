import torch
import torch.nn as nn 
import torch.nn.functional as F
eps = 1e-8


def gan_loss(output, label):
    target = torch.full_like(output, label)
    loss = F.mse_loss(output, target)
    return loss

def kld_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def DAMSM_loss(attn_score, batch_sum, gamma_3):
    loss = torch.exp(gamma_3 * attn_score)/batch_sum
    return loss

def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt
    