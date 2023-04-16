import os
import argparse

import torch
from scipy.io import wavfile

import jukebox
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.hparams import Hyperparams
from jukebox.sample import make_model, load_prompts
from jukebox.utils.audio_utils import save_wav
import utils.globals as uglobals

# Globals
MODEL = '1b_lyrics'
STEP_SIZE = 45000

def run_vqvaes(dir, out_dir):
    # Set up devices
    rank, local_rank, device = setup_dist_from_mpi(port=29500)

    # Set up model hps for inference
    hps = Hyperparams(
        name = 'sample_1b',
        levels = 3,
        sample_length_in_seconds = 20,
        total_sample_length_in_seconds = 180,
        sr = 44100,
        n_samples = 16,
        hop_fraction = [0.5, 0.5, 0.125]
    )

    # Load the models
    vqvae, priors = make_model(MODEL, device, hps)

    # Throw all the prior models to the cpu
    for prior in priors:
        prior.prior.cpu()

    if not os.path.exists(f'{out_dir}/z'):    
        os.makedirs(f'{out_dir}/z')
    if not os.path.exists(f'{out_dir}/recons'):    
        os.makedirs(f'{out_dir}/recons')

    for file_name in os.listdir(dir):
        sample_path = f'{dir}/{file_name}'
        save_name_z = f'{out_dir}/z/{file_name}'.replace('wav', 'pt')
        save_name_recons = f'{out_dir}/recons/{file_name}'

        with torch.no_grad():
            for prior_lv, prior in enumerate(priors):
                print(prior_lv)
                
                sr, data = wavfile.read(sample_path)

                raw_to_tokens = prior.raw_to_tokens
                duration = (int(len(data)) // raw_to_tokens) * raw_to_tokens
                x = load_prompts([sample_path], duration, hps)

                i = 0
                z, x_recons = [], []

                # If we have enough vram
                z = prior.encode(x, bs_chunks=x.shape[0])
                x_recons = prior.decode(z, bs_chunks=z[prior_lv].shape[0])

                # while i <= x.shape[1]:
                #     print(i)
                
                #     x_step = x[:, i: i + STEP_SIZE, :]

                #     z_step = prior.encode(x_step, bs_chunks=x.shape[0])
                #     x_recons_step = prior.decode(z_step, bs_chunks=z_step[prior_lv].shape[0])

                #     i += STEP_SIZE

                #     # Piece the steps together
                #     for i, val in enumerate(z_step):
                #         z_step[i] = val.cpu()
                #     for i, val in enumerate(x_recons_step):
                #         x_recons_step[i] = val.cpu()

                #     if z == []:
                #         z = z_step
                #         x_recons = x_recons_step
                #         print(len(z_step))
                #         print(z_step[0].shape)
                #         print(len(x_recons_step))
                #         print(x_recons_step[0].shape)
                #     else:
                #         for i, val in enumerate(z):
                #             z[i] = torch.cat((val, z_step[i]), 0)
                #         for i, val in enumerate(x_recons):
                #             x_recons[i] = torch.cat((val, x_recons_step[i]), 1)

                torch.save(z, save_name_z)
                save_wav(save_name_recons, x_recons, hps.sr)

if __name__ == '__main__':
    run_vqvaes(f'{uglobals.MUSDB18_PROCESSED_PATH}/train/vocals', f'{uglobals.MUSDB18_ORACLE}')
    run_vqvaes(f'{uglobals.MUSDB18_PROCESSED_PATH}/train/accompaniment', f'{uglobals.MUSDB18_ORACLE}')
    run_vqvaes(f'{uglobals.MUSDB18_PROCESSED_PATH}/test/vocals', f'{uglobals.MUSDB18_ORACLE}')
    run_vqvaes(f'{uglobals.MUSDB18_PROCESSED_PATH}/test/accompaniment', f'{uglobals.MUSDB18_ORACLE}')