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
import matplotlib.pyplot as plt

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
MODEL_PATH = 'out_autoregressive.pt'
model = torch.load(MODEL_PATH).eval().cuda()

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
validation_data = RobomimicDataloader(dataset, train=False,
    context_length=seq_len-1, step_size=5)
batch_size = 8
train_loader, val_loader = data_loaders(
    training_data, validation_data, batch_size)

with torch.no_grad():
    for step, (img, idx) in enumerate(train_loader):
        if step >= 10:
            break
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
   
        out = out.view(seq_len, bsz*576, -1)
        out = out[-1]
        out_idxs = torch.argmax(out, dim=-1, keepdim=True)
        z_q = network.vector_quantization.recover_embeddings(out_idxs)
        x_hat = network.decoder(z_q)
        x_hat = x_hat.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
   
        img = img.view(bsz, context, *image_dim)
        img = img.permute(0, 1, 3, 4, 2) 
        img = img.detach().cpu().numpy()
        fig, ax = plt.subplots(1, seq_len)
        for i in range(seq_len - 1):
            ax[i].imshow((img[0][i] * 255.0).astype(np.uint8))
        ax[seq_len-1].imshow((x_hat* 255.0).astype(np.uint8))
        plt.savefig(f'predictions/prediction_{step}.png')    
        

