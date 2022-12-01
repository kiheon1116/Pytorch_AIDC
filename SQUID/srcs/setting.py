import os
import random
import argparse

import torch
import numpy as np

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch',          default=64, type=int)
    parser.add_argument('--ber_stride',     default=10, type=int)
    parser.add_argument('--dset_path',      default='/media/1/ImageNet/val')
    parser.add_argument('--g_seed',         default=0,  type=int)
    parser.add_argument('--iteration',      default=10, type=int)
    parser.add_argument('--iteration_start',default=0,  type=int)
    parser.add_argument('--l_seed',         default=0,  type=int)
    parser.add_argument('--num_workers',    default=8, type=int)
    parser.add_argument('--save_path',      default='../result')
    parser.add_argument('--weight',         default='DEFAULT')
    
    parser.add_argument('--ber',            default=-5, type=float)    
    parser.add_argument('--model',          default='resnet50')
    parser.add_argument('--nbit',           default=8, type=int)
    parser.add_argument('--option',         default='default')
    parser.add_argument('--quant',          default='channel')
    
    
    return parser.parse_args()

def manual_l_seed():
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def set_environment(args):

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.g_seed + int((-args.ber) * 100))

    for _ in range(args.iteration_start):
        random.randint(0, 2**32 - 1)
