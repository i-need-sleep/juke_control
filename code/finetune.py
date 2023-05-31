import os
import sys
import argparse
import random

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
import utils.globals as uglobals

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(21) 

def finetune(args, dist_setup=None):
    if args.debug:
        args.batch_size = 1

    print(args)

    # Set up devices
    if dist_setup == None:
        rank, local_rank, device = setup_dist_from_mpi(port=29500)
    else:
        rank, local_rank, device = dist_setup

    # Set up hyperparameters
    kwargs = {
        'name': args.name,
        'local_logdir': uglobals.CHECKPOINT_DIR,
        'sample_length': 1048576,
        'bs': args.batch_size,
        'aug_shift': True,
        'aug_blend': True,
        'labels': True,
        'train': True,
        'test': True,
        'prior': True,
        'levels': 3,
        'level': 2,
        'weight_decay': 0.01,
        'save_iters': 1000,
        'lr': args.lr
    }
    # Restore pre-trained/finetuned checkpoint
    if args.checkpoint != '':
        kwargs['restore_prior'] = args.checkpoint
    # LR decay
    if args.lr_start_linear_decay != -1:
        kwargs['lr_start_linear_decay'] = args.lr_start_linear_decay
    if args.lr_decay != -1:
        kwargs['lr_decay'] = args.lr_decay
    hps = setup_hparams('vqvae,prior_1b_lyrics,all_fp16,cpu_ema', kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs

    hps.strict = not args.eval # Allow adding new params for ft
    hps.controlnet = args.controlnet

    # Setup models
    vqvae = make_vqvae(hps, device)
    print_once(f"Parameters VQVAE:{count_parameters(vqvae)}")
    prior = make_prior(hps, vqvae, device, debug=args.debug)
    print_once(f"Parameters Prior:{count_parameters(prior)}")
    model = prior

    # Setup opt, ema and distributed_model.
    opt, shd, scalar = get_optimizer(model, hps)
    ema = get_ema(model, hps)
    distributed_model = get_ddp(model, hps)
    
    # Make datasets
    if args.dataset == 'musdb18':
        dir = uglobals.MUSDB18_ORACLE
    elif args.dataset == 'urmp':
        dir = uglobals.URMP_ORACLE
    else:
        raise NotImplementedError
    train_loader = dataset.build_z2z_loader(f'{dir}/train/{args.src}/z', f'{dir}/train/{args.tar}/z', hps.bs, controlnet=hps.controlnet)
    dev_loader = dataset.build_z2z_loader(f'{dir}/dev/{args.src}/z', f'{dir}/dev/{args.tar}/z', 1, random_offset=False, shuffle=False, controlnet=hps.controlnet)
    test_loader = dataset.build_z2z_loader(f'{dir}/test/{args.src}/z', f'{dir}/test/{args.tar}/z', 1, random_offset=False, shuffle=False, controlnet=hps.controlnet)

    if args.eval:
        if args.eval_on_train:
            loader = train_loader
        else:
            loader = test_loader
        if args.controlnet:
            save_dir = eval_controlnet(model, loader, hps, args)
        else:
            save_dir = eval(model, loader, hps, args)
        return save_dir
    
    # Loggers
    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Train
    for epoch in range(hps.curr_epoch, hps.epochs):
        metrics.reset()
        train_loader.dataset.slice_data()
        if hps.train:
            if hps.controlnet:
                train_metrics = train_controlnet(distributed_model, model, opt, shd, scalar, ema, logger, metrics, train_loader, hps, args)
            else:
                train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, train_loader, hps, args)
            train_metrics['epoch'] = epoch
            if rank == 0:
                print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
            dist.barrier()

        if epoch % args.eval_epoch_interval == 0:
            # Eval for test loader loss
            dev_loader.dataset.slice_data()
            with torch.no_grad():
                if hps.controlnet:
                    losses = train_controlnet(distributed_model, model, opt, shd, scalar, ema, logger, metrics, dev_loader, hps, args, eval=True)
                else:
                    losses = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, dev_loader, hps, args, eval=True)
            if rank == 0:
                avg_loss = sum(losses) / len(losses)
                print(f'Dev eval epoch: {epoch}, loss: {avg_loss}')
                logger.add_scalar('Dev loss', avg_loss)
            dist.barrier()
        dist.barrier()
    
    return

def train(model, orig_model, opt, shd, scalar, ema, logger, metrics, loader, hps, args, eval=False):
    model.train()
    orig_model.train()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd", gn="gn", g_l="gen_loss", p_l="prime_loss")
    else:
        _print_keys = dict(l="loss", sl="spectral_loss", rl="recons_loss", e="entropy", u="usage", uc="used_curr", gn="gn", pn="pn", dk="dk")

    losses = []

    for batch_idx, batch in logger.get_range(loader):
        
        # Unpack batch
        z = batch['z'].to('cuda', non_blocking=True).long()
        pred_mask = batch['pred_mask'].to('cuda', non_blocking=True)
        sep_mask = batch['sep_mask'].to('cuda', non_blocking=True)
        pad_mask = batch['pad_mask'].to('cuda', non_blocking=True)
        song_name = batch['song_name']
        start = batch['start']
        total = batch['total']

        # Build y with default artist/style/lyrics conditions
        raw_to_tokens = orig_model.raw_to_tokens
        for i in range(z.shape[0]):
            label = orig_model.labeller.get_label('unknown', 'unknown', '', total[i]*raw_to_tokens, start[i]*raw_to_tokens) # duration (sr), offset within song  
            if i == 0:
                y = torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)
            else:
                y = torch.cat((y, torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)), dim=0)

        log_input_output = (logger.iters % hps.save_iters == 0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
            
        # Forward
        x_out, loss, _metrics = orig_model.finetune_forward(z, pred_mask, sep_mask, pad_mask, **forw_kwargs)

        if eval:
            losses.append(float(loss.detach().cpu()))
            continue

        # Backward
        loss, scale, grad_norm, overflow_loss, overflow_grad = backward(loss=loss, params=list(model.parameters()),
                                                                        scalar=scalar, fp16=hps.fp16, logger=logger)

        # Skip step if overflow
        grad_norm = allreduce(grad_norm, op=dist.ReduceOp.MAX)
        if overflow_loss or overflow_grad or grad_norm > hps.ignore_grad_norm > 0:
            zero_grad(orig_model)
            continue

        # Step opt. Divide by scale to include clipping and fp16 scaling
        logger.step()
        opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        zero_grad(orig_model)
        lr = hps.lr if shd is None else shd.get_lr()[0]
        if shd is not None: shd.step()
        if ema is not None: ema.step()
        next_lr = hps.lr if shd is None else shd.get_lr()[0]
        finished_training = (next_lr == 0.0)

        # Logging
        for key, val in _metrics.items():
            _metrics[key] = val.item()
        _metrics["loss"] = loss = loss.item() * hps.iters_before_update # Make sure to call to free graph
        _metrics["gn"] = grad_norm
        _metrics["lr"] = lr
        _metrics["lg_loss_scale"] = np.log2(scale)

        # Average and log
        for key, val in _metrics.items():
            _metrics[key] = metrics.update(key, val, z.shape[0])
            if logger.iters % hps.log_steps == 0:
                logger.add_scalar(key, _metrics[key])

        # Save checkpoint
        with torch.no_grad():
            if hps.save and (logger.iters % hps.save_iters == 1 or finished_training) and (logger.iters > 5000 or 'results/checkpoints' in hps.restore_prior):
                if ema is not None: ema.swap()
                orig_model.eval()
                name = f'step_{logger.iters}'
                if dist.get_rank() % 8 == 0:
                    print('Saving')
                    save_checkpoint(logger, name, orig_model, opt, dict(step=logger.iters), hps)
                orig_model.train()
                if ema is not None: ema.swap()

        logger.set_postfix(**{print_key:_metrics[key] for print_key, key in _print_keys.items()})
        if finished_training:
            dist.barrier()
            exit()
    
    if eval:
        return losses
            
    logger.close_range()
    return {key: metrics.avg(key) for key in _metrics.keys()}

def eval(model, loader, hps, args):
    model.eval()

    try:
        eval_size = args.eval_size
    except:
        eval_size = 5

    if args.dataset == 'musdb18':
        save_dir = f'{uglobals.MUSDB18_Z_OUT}/{hps.name}'
    elif args.dataset == 'urmp':
        save_dir = f'{uglobals.URMP_Z_OUT}/{hps.name}'
    else:
        raise NotImplemented
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_pred, n_hit = 0, 0

    if args.dataset == 'musdb18':
        indices = uglobals.musdb18_finetune_indices[: eval_size]
    else:
        indices = [3, 10, 21]
    for batch_idx, batch in enumerate(loader):
        if batch_idx not in indices:
            continue
        
        # bs is always 1
        # Unpack batch
        z = batch['z'].to('cuda', non_blocking=True).long()
        pred_mask = batch['pred_mask'].to('cuda', non_blocking=True)
        sep_mask = batch['sep_mask'].to('cuda', non_blocking=True)
        pad_mask = batch['pad_mask'].to('cuda', non_blocking=True)
        song_name = batch['song_name']
        start = batch['start']
        total = batch['total']

        # Build y with default artist/style/lyrics conditions
        raw_to_tokens = model.raw_to_tokens
        for i in range(z.shape[0]):
            label = model.labeller.get_label('unknown', 'unknown', '', total[i]*raw_to_tokens, start[i]*raw_to_tokens) # duration (sr), offset within song  
            if i == 0:
                y = torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)
            else:
                y = torch.cat((y, torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)), dim=0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

        # Sample z sequence
        z_pred, z_true = model.finetune_sample_z(z, pred_mask, sep_mask, pad_mask, **forw_kwargs)
        
        # Calculate accuracy
        n_pred += z_pred.shape[1]
        n_hit += torch.sum(z_pred == z_true).int()

        # Save z
        save_path = f'{save_dir}/{song_name[0]}_{start[0]}_{total[0]}.pt'
        torch.save({
            'z_pred': z_pred,
            'z_true': z_true,
        }, save_path)

        # Evaluate acc
        print(f'Overall accuracy: {n_hit / n_pred}, n_pred: {n_pred}, n_hit: {n_hit}')
    return save_dir

def train_controlnet(model, orig_model, opt, shd, scalar, ema, logger, metrics, loader, hps, args, eval=False):

    # Build a list of the parameters to train    
    params_to_train = []
    for name, param in orig_model.named_parameters():
        if 'prior.' not in name:
            params_to_train.append(param)

    model.train()
    orig_model.train()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd", gn="gn", g_l="gen_loss", p_l="prime_loss")
    else:
        _print_keys = dict(l="loss", sl="spectral_loss", rl="recons_loss", e="entropy", u="usage", uc="used_curr", gn="gn", pn="pn", dk="dk")

    losses = []

    for batch_idx, batch in logger.get_range(loader):
        
        # Unpack batch
        z_src = batch['z_src'].to('cuda', non_blocking=True).long()
        z_tar = batch['z_tar'].to('cuda', non_blocking=True).long()
        pred_mask = batch['pred_mask'].to('cuda', non_blocking=True)
        pad_mask = batch['pad_mask'].to('cuda', non_blocking=True)
        song_name = batch['song_name']
        start = batch['start']
        total = batch['total']

        # Build y with default artist/style/lyrics conditions
        raw_to_tokens = orig_model.raw_to_tokens
        for i in range(z_src.shape[0]):
            label = orig_model.labeller.get_label('unknown', 'unknown', '', total[i]*raw_to_tokens, start[i]*raw_to_tokens) # duration (sr), offset within song  
            if i == 0:
                y = torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)
            else:
                y = torch.cat((y, torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)), dim=0)

        log_input_output = (logger.iters % hps.save_iters == 0)
        
        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
            
        # Forward
        x_out, loss, _metrics = orig_model.controlnet_forward(z_src, z_tar, pred_mask, pad_mask, **forw_kwargs)

        if eval:
            losses.append(float(loss.detach().cpu()))
            continue

        # Backward
        loss, scale, grad_norm, overflow_loss, overflow_grad = backward(loss=loss, params=params_to_train,
                                                                        scalar=scalar, fp16=hps.fp16, logger=logger)

        # Skip step if overflow
        grad_norm = allreduce(grad_norm, op=dist.ReduceOp.MAX)
        if overflow_loss or overflow_grad or grad_norm > hps.ignore_grad_norm > 0:
            zero_grad(orig_model)
            continue

        # Step opt. Divide by scale to include clipping and fp16 scaling
        logger.step()
        opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        zero_grad(orig_model)
        lr = hps.lr if shd is None else shd.get_lr()[0]
        if shd is not None: shd.step()
        if ema is not None: ema.step()
        next_lr = hps.lr if shd is None else shd.get_lr()[0]
        finished_training = (next_lr == 0.0)

        # Logging
        for key, val in _metrics.items():
            _metrics[key] = val.item()
        _metrics["loss"] = loss = loss.item() * hps.iters_before_update # Make sure to call to free graph
        _metrics["gn"] = grad_norm
        _metrics["lr"] = lr
        _metrics["lg_loss_scale"] = np.log2(scale)

        # Average and log
        for key, val in _metrics.items():
            _metrics[key] = metrics.update(key, val, z_tar.shape[0])
            if logger.iters % hps.log_steps == 0:
                logger.add_scalar(key, _metrics[key])

        # Save checkpoint
        with torch.no_grad():
            if hps.save and (logger.iters % hps.save_iters == 1 or finished_training) and (logger.iters > 5000 or 'results/checkpoints' in hps.restore_prior):
                if ema is not None: ema.swap()
                orig_model.eval()
                name = f'step_{logger.iters}'
                if dist.get_rank() % 8 == 0:
                    print('Saving')
                    save_checkpoint(logger, name, orig_model, opt, dict(step=logger.iters), hps)
                orig_model.train()
                if ema is not None: ema.swap()

        logger.set_postfix(**{print_key:_metrics[key] for print_key, key in _print_keys.items()})
        if finished_training:
            dist.barrier()
            exit()

    if eval:
        return losses
            
    logger.close_range()
    return {key: metrics.avg(key) for key in _metrics.keys()}

def eval_controlnet(model, loader, hps, args):
    model.eval()

    try:
        eval_size = int(args.eval_size)
    except:
        eval_size = 5

    if args.dataset == 'musdb18':
        save_dir = f'{uglobals.MUSDB18_Z_OUT}/{hps.name}'
    elif args.dataset == 'urmp':
        save_dir = f'{uglobals.URMP_Z_OUT}/{hps.name}'
    else:
        raise NotImplemented
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_pred, n_hit = 0, 0

    if args.dataset == 'musdb18':
        indices = uglobals.musdb18_finetune_indices[: eval_size]
    elif args.dataset == 'urmp':
        indices = uglobals.urmp_controlnet_indices[: eval_size]
    else:
        indices = [3, 10, 21, ]
    for batch_idx, batch in enumerate(loader):
        if batch_idx not in indices:
            continue

        # bs is always 1
        # Unpack batch
        z_src = batch['z_src'].to('cuda', non_blocking=True).long()
        z_tar = batch['z_tar'].to('cuda', non_blocking=True).long()
        pred_mask = batch['pred_mask'].to('cuda', non_blocking=True)
        pad_mask = batch['pad_mask'].to('cuda', non_blocking=True)
        song_name = batch['song_name']
        start = batch['start']
        total = batch['total']

        # Build y with default artist/style/lyrics conditions
        raw_to_tokens = model.raw_to_tokens
        for i in range(z_src.shape[0]):
            label = model.labeller.get_label('unknown', 'unknown', '', total[i]*raw_to_tokens, start[i]*raw_to_tokens) # duration (sr), offset within song  
            if i == 0:
                y = torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)
            else:
                y = torch.cat((y, torch.tensor(label['y']).reshape(1, -1).to('cuda', non_blocking=True)), dim=0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

        # Sample z sequence
        z_pred, z_true = model.controlnet_sample_z(z_src, z_tar, pred_mask, pad_mask, **forw_kwargs)
        
        # Calculate accuracy
        n_pred += z_pred.shape[1]
        n_hit += torch.sum(z_pred == z_true).int()

        # Save z
        save_path = f'{save_dir}/{song_name[0]}_{start[0]}_{total[0]}.pt'
        torch.save({
            'z_pred': z_pred,
            'z_true': z_true,
        }, save_path)

        # Evaluate acc
        print(f'Overall accuracy: {n_hit / n_pred}, n_pred: {n_pred}, n_hit: {n_hit}')
    return save_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed', type=str) 
    parser.add_argument('--checkpoint', default='', type=str)

    # Model setup
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
    parser.add_argument('--eval_epoch_interval', default='1', type=int)

    # Eval
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_on_train', action='store_true')

    args = parser.parse_args()

    finetune(args)