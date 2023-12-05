import os
os.environ['WANDB_API_KEY'] = '6bc5ed6cff29e874c30c04f4a29faae7ae603964'
os.environ['WANDB_ENTITY'] = 'jyang27'
os.environ["WANDB__SERVICE_WAIT"] = "10000"
import numpy as np
import argparse
import pickle as pkl
import torch
import cv2
from torch.utils.data import DataLoader
from vqvae.datasets.robomimic import (RobomimicDataset,
    RobomimicDataloader, DatasetConfig)
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

DATA_PATH = '/iris/u/jyang27/dev/vqvae/data/square/processed_data96.hdf5'

parser = argparse.ArgumentParser()
parser.add_argument('--vqvae', type=str, required=True)
args = parser.parse_args()

network = torch.load(args.vqvae)
embedding_dim = network.vector_quantization.e_dim
n_codes = network.vector_quantization.n_e

d_model = 576  # Model dimension
nhead = 8     # Number of attention heads
num_layers = 6  # Number of decoder layers
dim_feedforward = 2048  # Feedforward dimension

logger = wandb.init(
    name='vqvae_autoregressive',
)


class AutoRegressiveModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, 
            embedding_dim, n_codes, seq_len):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_layers,
        ).cuda()
        self.fc_in = nn.Linear(embedding_dim, d_model, bias=False).cuda()
        self.fc_in.weight.data.normal_(std=0.02)
        self.fc_out = nn.Linear(d_model, n_codes, bias=False).cuda()
        self.mask = self.generate_square_subsequent_mask(seq_len, 'cuda')


    def forward(self, targets, memory):
        out = self.fc_out(self.decoder(targets, memory,
            tgt_mask=self.mask))
        return out

    def generate_square_subsequent_mask(self, sz: int, device: str = "cpu") -> torch.Tensor:
        """ Generate the attention mask for causal decoding """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        ).to(device=device)
        return mask


# Create the decoder
seq_len = 5
model = AutoRegressiveModel(d_model, nhead, dim_feedforward, num_layers,
            embedding_dim, n_codes, seq_len).cuda()

#Configure optimizer
optimizer = torch.optim.Adam(model.parameters(),
    lr=1e-4, amsgrad=True)

def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


cfg = DatasetConfig(path=DATA_PATH)
dataset = RobomimicDataset(cfg)
training_data = RobomimicDataloader(dataset, train=True, 
    context_length=seq_len-1, step_size=5)
validation_data = RobomimicDataloader(dataset, train=False)
batch_size = 8
train_loader, val_loader = data_loaders(
    training_data, validation_data, batch_size)

for step, (img, idx) in enumerate(train_loader):
    img = img.cuda()
    bsz, context, *image_dim  = img.shape
    img = img.view(bsz*context, *image_dim)
    z_e = network.encoder(img)
    z_e = network.pre_quantization_conv(z_e)
    embedding_loss, z_q, perplexity, _, min_encoding_indices = network.vector_quantization(
        z_e)
    embedding_idx = min_encoding_indices.view(bsz, context, -1)
    embedding_idx = embedding_idx.permute(1, 0, 2).contiguous()
    z_q = z_q.view(bsz, context, embedding_dim, 24, 24)
    z_q = z_q.permute(1, 0, 3, 4, 2).contiguous()
    targets = model.fc_in(z_q)
    targets = targets.view(context, bsz * 576, -1)

    #Add start embedding
    start_token = torch.zeros(targets.shape[1:]).unsqueeze(0).cuda()
    targets = torch.cat((start_token, targets[:-1]), dim=0)
    memory = torch.zeros(*targets.shape).cuda()
    out = model(targets, memory)

    label = embedding_idx.view(-1)
    out = out.view(label.shape[0], -1)
    loss = F.cross_entropy(out, label, reduction='none').mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.log(data={'Train Cross Entropy': loss}, step=step)

    if step % 100 == 0:
        torch.save(model, 'out_autoregressive.pt')


