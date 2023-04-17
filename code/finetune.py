import os
import sys
import argparse

import torch

from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.hparams import setup_hparams
import jukebox.utils.dist_adapter as dist
from jukebox.utils.dist_utils import print_once, allreduce, allgather
from jukebox.make_models import make_vqvae, make_prior, restore_opt, save_checkpoint
from jukebox.utils.torch_utils import zero_grad, count_parameters
from jukebox.train import get_ema, get_optimizer, get_ddp
from jukebox.utils.logger import init_logging

import utils.globals as uglobals

def finetune(args):
    print(args)

    # Set up devices
    rank, local_rank, device = setup_dist_from_mpi(port=29500)

    # Set up hyperparameters
    kwargs = {
        'name': args.name,
        'sample_length': 1048576,
        'bs': 1,
        'aug_shift': True,
        'aug_blend': True,
        'labels': True,
        'train': True,
        'test': True,
        'prior': True,
        'levels': 3,
        'level': 2,
        'weight_decay': 0.01,
        'save_iters': 1000
    }
    hps = setup_hparams('vqvae,prior_1b_lyrics,all_fp16,cpu_ema', kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs

    # Setup models
    vqvae = make_vqvae(hps, device)
    print_once(f"Parameters VQVAE:{count_parameters(vqvae)}")
    prior = make_prior(hps, vqvae, device)
    print_once(f"Parameters Prior:{count_parameters(prior)}")
    model = prior

    # Setup opt, ema and distributed_model.
    opt, shd, scalar = get_optimizer(model, hps)
    ema = get_ema(model, hps)
    distributed_model = get_ddp(model, hps)

    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Train
    for epoch in range(hps.curr_epoch, hps.epochs):
    metrics.reset()
    data_processor.set_epoch(epoch)
    if hps.train:
        train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, data_processor, hps)
        train_metrics['epoch'] = epoch
        if rank == 0:
            print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
        dist.barrier()

    if hps.test:
        if ema: ema.swap()
        test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
        test_metrics['epoch'] = epoch
        if rank == 0:
            print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
        dist.barrier()
        if ema: ema.swap()
    dist.barrier()
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed', type=str) 

    args = parser.parse_args()

    finetune(args)