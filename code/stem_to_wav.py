import os

import numpy as np
import tqdm
import stempeg

import utils.globals as uglobals

tracks = ['mix', 'drums', 'bass', 'other', 'vocals', 'accompaniment']
splits = ['train', 'test']

for split in splits:
    for file_name in tqdm.tqdm(os.listdir(f'{uglobals.MUSDB18_PATH}/{split}')):
        file_path = f'{uglobals.MUSDB18_PATH}/{split}/{file_name}'

        for stem_id in range(6):
            if stem_id < 5:
                s, rate = stempeg.read_stems(file_path, stem_id=[stem_id])
            else:
                s, rate = stempeg.read_stems(file_path, stem_id=[1, 2, 3])
                # Merge the tracks
                s = np.sum(s, axis=0)

            # Downmix the channels
            s = np.sum(s, axis = 1)

            save_dir = f'{uglobals.MUSDB18_PROCESSED_PATH}/{split}/{tracks[stem_id]}'

            if not os.path.exists(save_dir):    
                os.makedirs(save_dir)

            # Split songs into chunks of < 1 min
            i = 0
            chunk_size = rate * 60
            while i <= s.shape[0]:
                s_chunk = s[i: i + chunk_size]

                stempeg.write_audio(path=f'{save_dir}/{file_name}'.replace('.stem.mp4', f'_{int(i/chunk_size)}.wav'), data=s_chunk, sample_rate=rate, output_sample_rate=rate)
                i += chunk_size