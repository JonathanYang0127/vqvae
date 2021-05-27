import numpy as np
import argparse
import pickle as pkl
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--vqvae', type=str, required=True)
parser.add_argument('--buffer', type=str, required=True)
parser.add_argument('--normalized', action='store_true', default=False)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

with open(args.buffer, 'rb') as f:
    data = np.load(f, allow_pickle=True)

if args.normalized:
    multiplier = 255.0
else:
    multiplier = 1.0

network = torch.load(args.vqvae)
img = data[0]['observations'][args.index]['image'].astype(np.float32) / multiplier
img = torch.FloatTensor(img[None])
img = np.transpose(img, (0, 3, 1, 2)).cuda()

z_e = network.encoder(img)
z_e = network.pre_quantization_conv(z_e)
embedding_loss, z_q, perplexity, _, min_encoding_indices = network.vector_quantization(
    z_e)
print(z_q.shape)

_, xhat, _ = network(img)

img = img.cpu().numpy()[0] * multiplier
xhat = xhat.detach().cpu().numpy()[0] * multiplier

img = np.minimum(img, 255.0)
xhat = np.minimum(xhat, 255.0)
img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
xhat = np.transpose(xhat, (1, 2, 0)).astype(np.uint8)
cv2.imwrite('in_{}.png'.format(args.index), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite('out_{}.png'.format(args.index), cv2.cvtColor(xhat, cv2.COLOR_RGB2BGR))

