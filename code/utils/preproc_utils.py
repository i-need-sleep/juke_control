import os
import random
import shutil
import math

import numpy as np
import tqdm
from scipy.io import wavfile

import utils.globals as uglobals

def make_musdb18_splits(dev_size=10):
    # Randomly select songs from the MUSDB18 train split to make a dev split
    if not os.path.exists(f'{uglobals.MUSDB18_PATH}/dev'):    
        os.makedirs(f'{uglobals.MUSDB18_PATH}/dev')

    for idx in range(dev_size):
        file_name = random.choice(os.listdir(f'{uglobals.MUSDB18_PATH}/train'))
        os.rename(f'{uglobals.MUSDB18_PATH}/train/{file_name}', f'{uglobals.MUSDB18_PATH}/dev/{file_name}')

def stem_to_wav():
    import stempeg
    tracks = ['mix', 'drums', 'bass', 'other', 'vocals', 'accompaniment']
    splits = ['train', 'dev', 'test']

    for split in splits:
        for file_name in tqdm.tqdm(os.listdir(f'{uglobals.MUSDB18_RAW_DIR}/{split}')):
            file_path = f'{uglobals.MUSDB18_RAW_DIR}/{split}/{file_name}'

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

def make_urmp_splits(dev_size=4, test_size=5):
    # Randomly select songs from the MUSDB18 train split to make a dev split
    for split in ['dev', 'test', 'train']:
        if not os.path.exists(f'{uglobals.URMP_RAW_DIR}/{split}'):    
            os.makedirs(f'{uglobals.URMP_RAW_DIR}/{split}')

    # Filter out folders
    folders = []
    for folder_name in os.listdir(f'{uglobals.URMP_RAW_DIR}/unprocessed'):
        if os.path.isdir(f'{uglobals.URMP_RAW_DIR}/unprocessed/{folder_name}') and 'READ' not in folder_name and 'Supplementary' not in folder_name:
            folders.append(folder_name)

    folder_names = random.sample(folders, dev_size)
    for folder_name in folder_names:
        shutil.copytree(f'{uglobals.URMP_RAW_DIR}/unprocessed/{folder_name}', f'{uglobals.URMP_RAW_DIR}/dev/{folder_name}')
        folders.remove(folder_name)

    folder_names = random.sample(folders, test_size)
    for folder_name in folder_names:
        shutil.copytree(f'{uglobals.URMP_RAW_DIR}/unprocessed/{folder_name}', f'{uglobals.URMP_RAW_DIR}/test/{folder_name}')
        folders.remove(folder_name)
    
    for folder_name in folders:
        shutil.copytree(f'{uglobals.URMP_RAW_DIR}/unprocessed/{folder_name}', f'{uglobals.URMP_RAW_DIR}/train/{folder_name}')
        

def midi_to_wav(velo=70):
    # Synthesize MIDI files with sine waves
    import pretty_midi
    import audiolazy

    for split in ['dev', 'test', 'train']:
        in_dir = f'{uglobals.URMP_RAW_DIR}/{split}'
        out_dir = f'{uglobals.URMP_PROCESSED_DIR}/sine/{split}'
        wav_out_dir = f'{uglobals.URMP_PROCESSED_DIR}/wav/{split}'
        for dir in [out_dir, wav_out_dir]:
            if not os.path.exists(dir):    
                os.makedirs(dir)

        for folder_name in os.listdir(in_dir):
            for f in os.listdir(f'{in_dir}/{folder_name}'):
                if f[-3:] == 'wav' and f[:5] == 'AuMix':
                    wav_name = f
        
            # Copy the original wav
            sr, wav = wavfile.read(f'{in_dir}/{folder_name}/{wav_name}')

            # Split songs into chunks of < 1 min
            i = 0
            chunk_size = sr * 60
            while i <= wav.shape[0]:
                s_chunk = wav[i: i + chunk_size]

                wavfile.write(f'{wav_out_dir}/{folder_name}_{int(i/chunk_size)}.wav', sr, s_chunk)
                i += chunk_size
            
            # Convert note annotations into midi
            pm = pretty_midi.PrettyMIDI(initial_tempo=60)
            inst = pretty_midi.Instrument(1)

            # Resolve the note level annotations 
            for file in os.listdir(f'{in_dir}/{folder_name}'):
                if f'Notes_' in file and file[0] != '.':
                    with open(f'{in_dir}/{folder_name}/{file}', 'r') as f:
                        lines = f.readlines()
            
                    # Set all velocities to a constant
                    # Map each note to have the ground-truth timings
                    for line in lines:
                        start, freq, dur = line.split('		')
                        start = float(start)
                        freq = float(freq)
                        dur = float(dur)
                        pitch = round(audiolazy.lazy_midi.freq2midi(freq))
                        note = pretty_midi.Note(start=start, end=start+dur, pitch=pitch, velocity=velo)
                        inst.notes.append(note)
            
            pm.instruments.append(inst)
            synthesized = pm.synthesize(fs=uglobals.SAMPLE_RATE, wave=np.sin)
            wavfile.write(f'{uglobals.DATA_DIR}/temp.wav', uglobals.SAMPLE_RATE ,synthesized)

            # Split songs into chunks of < 1 min
            i = 0
            chunk_size = uglobals.SAMPLE_RATE * 60
            while i <= synthesized.shape[0]:
                s_chunk = synthesized[i: i + chunk_size]

                wavfile.write(f'{out_dir}/{folder_name}_{int(i/chunk_size)}.wav', uglobals.SAMPLE_RATE, s_chunk)
                i += chunk_size
                
    return