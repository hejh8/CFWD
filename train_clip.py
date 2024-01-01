import sys
import ssl
import argparse
import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion
from math import sqrt
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torchvision
import torch.optim
import argparse
import numpy as np
import clip_loss as clip_loss
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import clip
import pyiqa
import shutil

task_name="train0"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='/home/ubuntu/project/CFGW/configs/LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='./pretrain_models/pretrain_model.pth.tar', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    parser.add_argument('--prompt_pretrain_dir', type=str, default= './pretrain_models/prompt_pretrain/init_prompt_pair.pth')
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--length_prompt', type=int, default=16)


    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)

    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Current device: {}".format(device))
    if torch.cuda.is_available():
       current_gpu = torch.cuda.current_device()
       gpu_name = torch.cuda.get_device_name(current_gpu)
       print("Current GPU: {} - {}".format(current_gpu, gpu_name))
    config.device = device

    torch.manual_seed(args.seed) 
    np.random.seed(args.seed)     
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)   
    torch.backends.cudnn.benchmark = True   
    

    # data loading
    print("Current dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)  


    diffusion = DenoisingDiffusion(args, config)

    diffusion.train(DATASET)

if __name__ == "__main__":
    main()
