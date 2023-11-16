import numpy as np
import argparse
import pickle as pkl
import torch
import cv2

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.misc.wx250_utils import add_data_to_buffer_real_robot, DummyEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vqvae', type=str, required=True)
parser.add_argument('--buffer', type=str, required=True)
parser.add_argument('--index-range', type=int, nargs=2, default=[0, 2])
args = parser.parse_args()


image_size = 64
expl_env = DummyEnv(image_size=image_size)

replay_buffer = ObsDictReplayBuffer(
    int(1E6),
    expl_env,
    observation_keys=['image']
)
add_data_to_buffer_real_robot(args.buffer, replay_buffer,
                   validation_replay_buffer=None,
                   validation_fraction=0.8)


network = torch.load(args.vqvae)
img = replay_buffer._obs['image'][np.arange(*args.index_range)].reshape(-1, 3, 64, 64)
img = torch.FloatTensor(img)
img = img.cuda()

z_e = network.encoder(img)
z_e = network.pre_quantization_conv(z_e)
embedding_loss, z_q, perplexity, _, min_encoding_indices = network.vector_quantization(
    z_e)

_, xhat, _ = network(img)

img = img.cpu().numpy() * 255.0
xhat = xhat.detach().cpu().numpy() * 255.0 

img = np.minimum(img, 255.0)
xhat = np.minimum(xhat, 255.0)

for index in range(*args.index_range):
    img_out = np.transpose(img[index], (1, 2, 0)).astype(np.uint8)
    xhat_out = np.transpose(xhat[index], (1, 2, 0)).astype(np.uint8)
    cv2.imwrite('in_{}.png'.format(index), cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite('out_{}.png'.format(index), cv2.cvtColor(xhat_out, cv2.COLOR_RGB2BGR))

