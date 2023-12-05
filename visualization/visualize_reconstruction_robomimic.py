import numpy as np
import argparse
import pickle as pkl
import torch
import cv2
from vqvae.datasets.robomimic import (RobomimicDataset,
    RobomimicDataloader, DatasetConfig)

DATA_PATH = '/iris/u/jyang27/dev/vqvae/data/square/processed_data96.hdf5'

parser = argparse.ArgumentParser()
parser.add_argument('--vqvae', type=str, required=True)
args = parser.parse_args()

cfg = DatasetConfig(path=DATA_PATH)
dataset = RobomimicDataset(cfg)
obs_key = cfg.rl_camera

img = dataset[0][10][obs_key].float() / 255.0
img = img.cuda().unsqueeze(0)

network = torch.load(args.vqvae)
z_e = network.encoder(img)
z_e = network.pre_quantization_conv(z_e)
embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = network.vector_quantization(
    z_e)
import pdb; pdb.set_trace()
_, xhat, _ = network(img)

img = img.cpu().numpy() * 255.0
xhat = xhat.detach().cpu().numpy() * 255.0 

img = np.minimum(img, 255.0)
xhat = np.minimum(xhat, 255.0)

img_out = np.transpose(img[0], (1, 2, 0)).astype(np.uint8)
xhat_out = np.transpose(xhat[0], (1, 2, 0)).astype(np.uint8)
cv2.imwrite('in.png', cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
cv2.imwrite('out.png', cv2.cvtColor(xhat_out, cv2.COLOR_RGB2BGR))

