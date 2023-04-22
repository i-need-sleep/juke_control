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

    for checkpoint_name in os.listdir(args.exp_name):
        # Test set prior inference
        args.eval_on_train = False
        args.checkpoint = f'{uglobals.CHECKPOINT_PATH}/{args.exp_name}/{checkpoint_name}'
        args.name = f'{args.exp_name}_{checkpoint_name}'

        # Decode to wav
        z_dir = finetune(args, dist_setup=dist_setup)
        wav_dir = z_dir.replace(uglobals.MUSDB18_Z_OUT, uglobals.MUSDB18_WAV_OUT)
        src_dir = f'{uglobals.MUSDB18_PROCESSED_PATH}/test/{args.tar}'
        dec(z_dir, src_dir, wav_dir, dist_setup)

        # Train set
        args.eval_on_train = True
        args.name = f'{args.exp_name}_{checkpoint_name}_train'
        save_dir = finetune(args, dist_setup=dist_setup)
        z_dir = finetune(args, dist_setup=dist_setup)

        wav_dir = z_dir.replace(uglobals.MUSDB18_Z_OUT, uglobals.MUSDB18_WAV_OUT)
        src_dir = f'{uglobals.MUSDB18_PROCESSED_PATH}/train/{args.tar}'
        dec(z_dir, src_dir, wav_dir, dist_setup)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--exp_name', default='', type=str)

    # Task
    parser.add_argument('--src', default='vocals', type=str) 
    parser.add_argument('--tar', default='acc', type=str) 

    # Training
    parser.add_argument('--batch_size', default='1', type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_start_linear_decay', default=-1, type=int) # already_trained_steps
    parser.add_argument('--lr_decay', default=-1, type=int) # decay_steps_as_needed

    args = parser.parse_args()

    eval_multiple(args)