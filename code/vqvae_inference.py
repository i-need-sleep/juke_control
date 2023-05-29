import os
import argparse
import math

import torch
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import jukebox
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.hparams import Hyperparams
from jukebox.sample import make_model, load_prompts
from jukebox.utils.audio_utils import save_wav
import utils.globals as uglobals

# Globals
MODEL = '1b_lyrics'

def enc_dec(dir, out_dir, dist_setup=None, controlnet=False):
    # Set up devices
    if dist_setup == None:
        rank, local_rank, device = setup_dist_from_mpi(port=29500)
    else:
        rank, local_rank, device = dist_setup

    # Set up model hps for inference
    hps = Hyperparams(
        name = 'sample_1b',
        levels = 3,
        sample_length_in_seconds = 20,
        total_sample_length_in_seconds = 180,
        sr = 44100,
        n_samples = 1,
        hop_fraction = [0.5, 0.5, 0.125]
    )
    hps.strict = False
    hps.controlnet = controlnet

    # Load the models
    vqvae, priors = make_model(MODEL, device, hps)

    # Throw all the prior models to the cpu
    for prior in priors:
        prior.prior.cpu()

    if not os.path.exists(f'{out_dir}/z'):    
        os.makedirs(f'{out_dir}/z')
    if not os.path.exists(f'{out_dir}/recons'):    
        os.makedirs(f'{out_dir}/recons')

    for file_name in tqdm(os.listdir(dir)):
        if not 'wav' in file_name:
            continue
        sample_path = f'{dir}/{file_name}'
        save_name_z = f'{out_dir}/z/{file_name}'.replace('.wav', '.pt')
        save_name_recons = f'{out_dir}/recons/{file_name}'

        with torch.no_grad():
            for prior_lv, prior in enumerate(reversed(priors)):
                
                sr, data = wavfile.read(sample_path)

                raw_to_tokens = prior.raw_to_tokens
                duration = (int(len(data)) // raw_to_tokens) * raw_to_tokens
                x = load_prompts([sample_path], duration, hps)
                
                i = 0
                z, x_recons = [], []

                z = prior.encode(x, bs_chunks=x.shape[0])
                x_recons = prior.decode(z, bs_chunks=z[prior_lv].shape[0])

                torch.save(z, save_name_z.replace('.pt', f'_{2-prior_lv}.pt'))
                if not os.path.exists(f'{save_name_recons}_{2-prior_lv}'):    
                    os.makedirs(f'{save_name_recons}_{2-prior_lv}')
                save_wav(f'{save_name_recons}_{2-prior_lv}', x_recons, hps.sr)

    return rank, local_rank, device

def dec(pred_dir, src_dir, tar_dir, out_dir, dist_setup=None, controlnet=False):
    # Set up devices
    if dist_setup == None:
        rank, local_rank, device = setup_dist_from_mpi(port=29500)
    else:
        rank, local_rank, device = dist_setup

    # Set up model hps for inference
    hps = Hyperparams(
        name = 'sample_1b',
        levels = 3,
        sample_length_in_seconds = 20,
        total_sample_length_in_seconds = 180,
        sr = 44100,
        n_samples = 1,
        hop_fraction = [0.5, 0.5, 0.125]
    )
    hps.strict = False
    hps.controlnet = controlnet

    # Load the models
    vqvae, priors = make_model(MODEL, device, hps)
    prior = priors[-1] # Top level prior

    for file_name in os.listdir(pred_dir):
        save_dir = f'{out_dir}/{file_name[:-3]}'
        if not os.path.exists(save_dir):    
            os.makedirs(save_dir)

        path = f'{pred_dir}/{file_name}'
        
        # Decode the prediction and oracle
        data = torch.load(path)
        z_pred = data['z_pred']
        z_true = data['z_true']
        
        x_pred = prior.decode([z_pred], bs_chunks=z_pred.shape[0])
        x_true = prior.decode([z_true], bs_chunks=z_pred.shape[0])

        if not os.path.exists(f'{save_dir}/pred'):    
            os.makedirs(f'{save_dir}/pred')
        if not os.path.exists(f'{save_dir}/oracle'):    
            os.makedirs(f'{save_dir}/oracle')
        save_wav(f'{save_dir}/pred', x_pred, hps.sr)
        save_wav(f'{save_dir}/oracle', x_true, hps.sr)

        # Retrieve the original vocal wav
        # URMP format: underscores in song names
        splits = file_name[:-3].split('_')
        start = splits[-2]
        total = splits[-1]
        song_name = '_'.join(splits[:-2])

        wav_root = f'{src_dir}/{song_name}'

        # Retrieve all pieces
        i = 0
        while True:
            wav_path = f'{wav_root}_{i}.wav'
            if not os.path.isfile(wav_path):
                break
            sr, data = wavfile.read(wav_path)
            data = data.reshape(1, -1)
            if i == 0:
                src_wav = data
            else:
                src_wav = np.concatenate((src_wav, data), axis=1)
            i += 1

        # Align the right slice
        start_idx = int(math.floor(src_wav.shape[1] / int(total) * int(start)))

        src_slice = src_wav[:, start_idx: start_idx + x_pred.shape[1]]
        src_slice = torch.tensor(src_slice).reshape(1, -1, 1).cuda() / 40000 # TODO: Check the scale 

        mix_pred = src_slice + x_pred
        mix_oracle = src_slice + x_true

        # Also include a slice from the target
        wav_root = f'{tar_dir}/{song_name}'

        # Retrieve all pieces
        i = 0
        while True:
            wav_path = f'{wav_root}_{i}.wav'
            if not os.path.isfile(wav_path):
                break
            sr, data = wavfile.read(wav_path)
            data = data.reshape(1, -1)
            if i == 0:
                tar_wav = data
            else:
                tar_wav = np.concatenate((tar_wav, data), axis=1)
            i += 1

        # Align the right slice
        start_idx = int(math.floor(tar_wav.shape[1] / int(total) * int(start)))

        tar_slice = tar_wav[:, start_idx: start_idx + x_pred.shape[1]]
        tar_slice = torch.tensor(tar_slice).reshape(1, -1, 1).cuda() / 40000 # TODO: Check the scale 
        
        # Save
        if not os.path.exists(f'{save_dir}/mix_pred'):    
            os.makedirs(f'{save_dir}/mix_pred')
        if not os.path.exists(f'{save_dir}/mix_oracle'):    
            os.makedirs(f'{save_dir}/mix_oracle')
        if not os.path.exists(f'{save_dir}/src'):    
            os.makedirs(f'{save_dir}/src')
        if not os.path.exists(f'{save_dir}/tar'):    
            os.makedirs(f'{save_dir}/tar')
        save_wav(f'{save_dir}/mix_pred', mix_pred, hps.sr)
        save_wav(f'{save_dir}/mix_oracle', mix_oracle, hps.sr)
        save_wav(f'{save_dir}/src', src_slice, hps.sr)
        save_wav(f'{save_dir}/tar', tar_slice, hps.sr)

    return

if __name__ == '__main__':
    dist_setup = None
    dist_setup = enc_dec(f'{uglobals.URMP_PROCESSED_DIR}/wav/train', f'{uglobals.URMP_ORACLE}/train/wav', dist_setup=dist_setup)
    dist_setup = enc_dec(f'{uglobals.URMP_PROCESSED_DIR}/sine/dev', f'{uglobals.URMP_ORACLE}/dev/sine', dist_setup=dist_setup)
    dist_setup = enc_dec(f'{uglobals.URMP_PROCESSED_DIR}/wav/dev', f'{uglobals.URMP_ORACLE}/dev/wav', dist_setup=dist_setup)
    dist_setup = enc_dec(f'{uglobals.URMP_PROCESSED_DIR}/sine/test', f'{uglobals.URMP_ORACLE}/test/sine', dist_setup=dist_setup)
    dist_setup = enc_dec(f'{uglobals.URMP_PROCESSED_DIR}/wav/test', f'{uglobals.URMP_ORACLE}/test/wav/', dist_setup=dist_setup)