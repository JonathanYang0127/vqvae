import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from vqvae.datasets.block import BlockDataset, LatentBlockDataset
from vqvae.datasets.widowx import WidowXDataset
import numpy as np


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val

def load_widowx():
    '''
    Loads sim datasets (uses rlkit from cql-private)
    '''
    from rlkit.data_management.load_buffer import load_data_from_npy
    import roboverse

    data_folder_path = '/home/jonathanyang0127/minibullet/data'
    data_file_path = data_folder_path + \
            ('/may25_Widow250OneObjectGraspShed-v0_20K_save_all_noise_0.1_2021-05-25T19-27-54/'
            'may25_Widow250OneObjectGraspShed-v0_20K_save_all_noise_0.1_2021-05-25T19-27-54_20000.npy')
    variant = {'buffer': data_file_path,
            'env': 'Widow250OneObjectGraspTrain-v0'}

    expl_env = roboverse.make(variant['env'], transpose_image=True)
    replay_buffer = load_data_from_npy(variant, expl_env, ['image'])
    train = WidowXDataset(replay_buffer, train=True, normalize=True, image_dims=(48, 48, 3))
    val = WidowXDataset(replay_buffer, train=False, normalize=True, image_dims=(48, 48, 3))

    return train, val

def load_widowx_real_robot():
    '''
    Loads sim datasets (uses rlkit from railrl-private)
    '''
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
    from rlkit.misc.wx250_utils import add_data_to_buffer_real_robot, DummyEnv

    image_size = 64
    expl_env = DummyEnv(image_size=image_size)
    data_folder_path = '/nfs/kun1/users/albert/realrobot_datasets'
    data_file_path = data_folder_path + '/combined_2021-05-20_21_53_31.pkl'

    replay_buffer = ObsDictReplayBuffer(
        int(1E6),
        expl_env,
        observation_keys=['image']
    )
    add_data_to_buffer_real_robot(data_file_path, replay_buffer,
                       validation_replay_buffer=None,
                       validation_fraction=0.8)

    train = WidowXDataset(replay_buffer, train=True, normalize=False, image_dims=(64, 64, 3))
    val = WidowXDataset(replay_buffer, train=False, normalize=False, image_dims=(64, 64, 3))

    return train, val



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


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        print(training_data.data.shape)
        print(next(iter(training_loader))[0].shape)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'WIDOWX':
        training_data, validation_data = load_widowx()
        print(training_data.data[0])
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'WIDOWXREALROBOT':
        training_data, validation_data = load_widowx_real_robot()
        print(training_data.data[0])
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')

def save_model_full(model, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    
    SAVE_MODEL_FILE = SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth'

    if not os.path.exists(SAVE_MODEL_FILE):
        with open(SAVE_MODEL_FILE, 'w+'):
            pass
    torch.save(model, SAVE_MODEL_FILE)

