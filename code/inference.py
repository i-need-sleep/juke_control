import os
import sys
import argparse

from scipy.io import wavfile
import torch
import numpy as np

from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.hparams import setup_hparams
import jukebox.utils.dist_adapter as dist
from jukebox.utils.dist_utils import print_once, allreduce, allgather
from jukebox.make_models import make_vqvae, make_prior, restore_opt, save_checkpoint
from jukebox.utils.torch_utils import zero_grad, count_parameters
from jukebox.train import get_ema, get_optimizer, get_ddp
from jukebox.utils.logger import init_logging
from jukebox.utils.io import get_duration_sec, load_audio
from jukebox.data.labels import Labeller
from jukebox.utils.fp16 import FP16FusedAdam, FusedAdam, LossScalar, clipped_grad_scale, backward
from jukebox.train import log_aud, log_labels

import dataset
from finetune import finetune
from vqvae_inference import dec
import utils.globals as uglobals

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(21) 

def eval_multiple(args):
    # For each file in the dir, evaluate on both training and test sets
    dist_setup = setup_dist_from_mpi(port=29500)
    args.eval = True

    if args.dataset == 'musdb18':
        processed_dir = uglobals.MUSDB18_PROCESSED_PATH
    elif args.dataset == 'urmp':
        processed_dir = uglobals.URMP_PROCESSED_DIR

    for checkpoint_name in os.listdir(f'{uglobals.CHECKPOINT_DIR}/{args.exp_name}'):
        if 'pth' not in checkpoint_name:
            continue
        # Test set prior inference
        args.eval_on_train = False
        args.checkpoint = f'{uglobals.CHECKPOINT_DIR}/{args.exp_name}/{checkpoint_name}'
        args.name = f'{args.exp_name}_{checkpoint_name}'

        # Decode to wav
        z_dir = finetune(args, dist_setup=dist_setup)
        # z_dir = '../results/outputs/musdb18/z_out/finetune_vocal2acc_checkpoint_step_15001.pth.tar'
        wav_dir = z_dir.replace(uglobals.MUSDB18_Z_OUT, uglobals.MUSDB18_WAV_OUT)
        src_dir = f'{processed_dir}/test/{args.src}'
        tar_dir = f'{processed_dir}/test/{args.tar}'
        dec(z_dir, src_dir, tar_dir, wav_dir, dist_setup, controlnet=args.controlnet)

        # Train set
        if not args.skip_train:
            args.eval_on_train = True
            args.name = f'{args.exp_name}_{checkpoint_name}_train'

            z_dir = finetune(args, dist_setup=dist_setup)
            wav_dir = z_dir.replace(uglobals.MUSDB18_Z_OUT, uglobals.MUSDB18_WAV_OUT)
            src_dir = f'{processed_dir}/train/{args.src}'
            tar_dir = f'{processed_dir}/train/{args.tar}'
            dec(z_dir, src_dir, tar_dir, wav_dir, dist_setup, controlnet=args.controlnet)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--controlnet', action='store_true')

    # Task
    parser.add_argument('--dataset', default='musdb18', type=str) 
    parser.add_argument('--src', default='vocals', type=str) 
    parser.add_argument('--tar', default='accompaniment', type=str) 

    # Training
    parser.add_argument('--batch_size', default='1', type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_start_linear_decay', default=-1, type=int) # already_trained_steps
    parser.add_argument('--lr_decay', default=-1, type=int) # decay_steps_as_needed

    # Eval
    parser.add_argument('--eval_size', default='3', type=int)
    parser.add_argument('--skip_train', action='store_true') # Skip eval on the train set

    args = parser.parse_args()

    eval_multiple(args)