import numpy as np
import argparse
import pickle as pkl
import torch
import cv2
from torch.utils.data import DataLoader
from vqvae.datasets.robomimic import (RobomimicDataset,
    RobomimicObsActionDataloader, DatasetConfig)
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_PATH = '/iris/u/jyang27/dev/vqvae/data/square/processed_data96.hdf5'

d_model = 576  # Model dimension
nhead = 8     # Number of attention heads
num_layers = 6  # Number of decoder layers
dim_feedforward = 2048  # Feedforward dimension

class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class ConvInverseDynamicsModel(nn.Module):
    def __init__(self, numChannels):   
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size=(3, 3), stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20,
            kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=2000, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=500, out_features=7)
        self.aug = RandomShiftsAug(4)

    def forward(self, x):
        x = self.aug(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

class PredictionInverseDynamicsModel(nn.Module):
    def __init__(self, vqvae_path, autoregressive_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        
        self.vqvae_encoder = torch.load(vqvae_path).eval().to(self.device)
        self.embedding_dim = self.vqvae_encoder.vector_quantization.e_dim
        self.n_codes = self.vqvae_encoder.vector_quantization.n_e

        self.prediction_model = torch.load(autoregressive_path).eval().to(self.device)
        self.seq_len = 2

        self.inverse_dynamics = ConvInverseDynamicsModel(
            6).to(self.device)

    def forward(self, img):
        #Image is batch, c, h, w
        with torch.no_grad():
            curr_embed, next_embed = self.get_current_and_next_embeddings(img)

        #Fuse embeddings along channel
        fusion = torch.cat((curr_embed, next_embed), dim=1)

        #Compute action
        action = self.inverse_dynamics(fusion)
        return action

    def get_current_and_next_embeddings(self, img):
        #Add context dim
        original_image = img
        img = img.unsqueeze(1)

        #Encode Image
        bsz, context, *image_dim  = img.shape
        img = img.view(bsz*context, *image_dim)
        z_e = self.vqvae_encoder.encoder(img)
        z_e = self.vqvae_encoder.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, min_encoding_indices = self.vqvae_encoder.vector_quantization(
            z_e)
        curr_z_q = z_q
        embedding_idx = min_encoding_indices.view(bsz, context, -1)
        embedding_idx = embedding_idx.permute(1, 0, 2).contiguous()
        z_q = z_q.view(bsz, context, self.embedding_dim, 24, 24)
        z_q = z_q.permute(1, 0, 3, 4, 2).contiguous()
        targets = self.prediction_model.fc_in(z_q)
        targets = targets.view(context, bsz * 576, -1)

        #Add start embedding
        start_token = torch.zeros(targets.shape[1:]).unsqueeze(0).to(self.device)
        targets = torch.cat((start_token, targets), dim=0)
        memory = torch.zeros(*targets.shape).to(self.device)

        #Get prediction (codebook index)
        out = self.prediction_model(targets, memory)
        out = out.view(self.seq_len, bsz *576, -1)
        out = out[-1]
        out_idxs = torch.argmax(out, dim=-1, keepdim=True)

        #Recover next embedding
        next_z_q = self.vqvae_encoder.vector_quantization.recover_embeddings(out_idxs)
        next_xhat = self.vqvae_encoder.decoder(next_z_q)
        curr_xhat =  self.vqvae_encoder.decoder(curr_z_q) 
        plt.imshow((next_xhat[0]* 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        plt.savefig('train_image.png')
        #return curr_z_q, next_z_q
        return original_image, next_xhat

class AutoRegressiveModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, 
            embedding_dim, n_codes, seq_len, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_layers,
        ).to(self.device)
        self.fc_in = nn.Linear(embedding_dim, d_model, bias=False).to(self.device)
        self.fc_in.weight.data.normal_(std=0.02)
        self.fc_out = nn.Linear(d_model, n_codes, bias=False).to(self.device)
        self.mask = self.generate_square_subsequent_mask(seq_len, device)


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

#Create data loader
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
training_data = RobomimicObsActionDataloader(dataset, train=True, 
    context_length=0, step_size=1)
validation_data = RobomimicObsActionDataloader(dataset, train=False,
    context_length=0, step_size=1)
batch_size = 64
train_loader, val_loader = data_loaders(
    training_data, validation_data, batch_size)

#Create model
#path = 'out_inverse_dynamics.pt'
#model = torch.load(path)
model = PredictionInverseDynamicsModel('vqvae_data_agentview.pth', 
    'out_autoregressive_seq_2_step_1.pt')

#Configure optimizer
optimizer = torch.optim.Adam(model.parameters(),
    lr=2e-4)


for epoch in range(20): 
    for step, (data, idx) in enumerate(train_loader):
        img = data['image']
        action = data['action']
        img = img.cuda()
        action = action.cuda()
        out = model(img)
        loss = F.mse_loss(out, action, reduction='none').mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(loss)
        if step % 100 == 0:
            torch.save(model, f'out_inverse_dynamics.pt')


