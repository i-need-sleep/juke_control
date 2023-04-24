import os
import random

import torch
from torch.utils.data import Dataset, DataLoader

import utils.globals as uglobals

class Z2ZDataset(Dataset):
    def __init__(self, src_dir, tar_dir, random_offset, window_len=3071, z_len=6144, controlnet=False):

        self.src_dir = src_dir
        self.tar_dir = tar_dir
        self.window_len = window_len # For each src/tar slice
        self.z_len = z_len # For model input
        self.random_offset = random_offset

        if controlnet:
            self.z_len *= 2

        self.sep = torch.zeros(1, 1) # Placeholders, will be replaced by learned tokens
        self.pad = torch.zeros(1, 1)
        
        self.prep_data()
        self.slice_data()

    def prep_data(self):
        # Build z sequences for each song
        pairs = {} # {song_name: [src, tar], ...} of [1, seq_len]
        for idx, file_name in enumerate(os.listdir(self.src_dir)):

            # Find song names
            if '_0_2.pt' not in file_name:
                continue
            song_name = file_name.replace('_0_2.pt', '')

            z_src = self._piece_z_parts(self.src_dir, song_name)
            z_tar = self._piece_z_parts(self.tar_dir, song_name) # [1, seq_len]

            pairs[song_name] = [z_src, z_tar]

        self.pairs = pairs

    def _piece_z_parts(self, dir, song_name):
        # Use only 2nd-level codes
        part_idx = 0
        zs = []

        while True:
            path = f'{dir}/{song_name}_{part_idx}_2.pt'
            if not os.path.isfile(path):
                break

            z = torch.load(path, map_location='cpu')[0]
            zs.append(z)
            part_idx += 1
        
        out = torch.cat(zs, dim=1) 
        return out # [1, seq_len]
    
    def slice_data(self):
        # Do this before each epoch
        # For each song, randomly choose a starting offset and slice z sequences to fit window_len
        # Piece the src and tar sequences into [src, SEP, tar, PAD]
        # Make masks for SEP and PAD

        out = [] # [{z, pred_mask, sep_mask, pad_mask, song_name, start, total}, ...] [1, seq_len]
        for song_name, [src, tar] in self.pairs.items():

            if self.random_offset:
                start = 0
            else:
                start = random.randint(0, self.window_len - 1)
            while start < src.shape[1]:
                src_slice = src[:, start: start + self.window_len]
                tar_slice = tar[:, start: start + self.window_len]

                z = torch.cat((src_slice, self.sep, tar_slice), dim=1)
                pred_mask = torch.cat((torch.zeros_like(src_slice), torch.zeros_like(self.sep), torch.ones_like(tar_slice)), dim=1)
                sep_mask = torch.cat((torch.zeros_like(src_slice), torch.ones_like(self.sep), torch.zeros_like(tar_slice)), dim=1)
                pad_mask = torch.zeros_like(sep_mask)
                
                # Pad
                while z.shape[1] < self.z_len:
                    z = torch.cat((z, self.pad), dim=1)
                    pred_mask = torch.cat((pred_mask, torch.zeros_like(self.pad)), dim=1)
                    sep_mask = torch.cat((sep_mask, torch.zeros_like(self.pad)), dim=1)
                    pad_mask = torch.cat((pad_mask, torch.ones_like(self.pad)), dim=1)

                out.append({
                    'z': z,
                    'pred_mask' : pred_mask,
                    'sep_mask': sep_mask,
                    'pad_mask': pad_mask,
                    'song_name': song_name,
                    'start': start, # In z length
                    'total': src.shape[1]
                })

                start += self.window_len

        self.slices = out
        return out
    
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        return self.slices[index]

def mr_collate(data):
    for line_idx, line in enumerate(data):
        if line_idx == 0:
            z = line['z']
            pred_mask = line['pred_mask']
            sep_mask = line['sep_mask']
            pad_mask = line['pad_mask']
            song_name = [line['song_name']]
            start = [line['start']]
            total = [line['total']]
        else:
            z  = torch.cat((z, line['z']), dim=0)
            pred_mask  = torch.cat((pred_mask, line['pred_mask']), dim=0)
            sep_mask  = torch.cat((sep_mask, line['sep_mask']), dim=0)
            pad_mask  = torch.cat((pad_mask, line['pad_mask']), dim=0)
            song_name.append(line['song_name'])
            start.append(line['start'])
            total.append(line['total'])
    return {
        'z': z,
        'pred_mask' : pred_mask,
        'sep_mask': sep_mask,
        'pad_mask': pad_mask,
        'song_name': song_name,
        'start': start,
        'total': total
    }

def ctrl_collate(data):
    for line_idx, line in enumerate(data):
        line = decouple_line(line)
        if line_idx == 0:
            z_src = line['z_src']
            z_tar = line['z_tar']
            src_pad_mask = line['src_pad_mask']
            tar_pad_mask = line['tar_pad_mask']
            tar_pred_mask = line['tar_pred_mask']
            song_name = [line['song_name']]
            start = [line['start']]
            total = [line['total']]
        else:
            z_src  = torch.cat((z_src, line['z_src']), dim=0)
            z_tar  = torch.cat((z_tar, line['z_tar']), dim=0)
            src_pad_mask  = torch.cat((src_pad_mask, line['src_pad_mask']), dim=0)
            tar_pad_mask  = torch.cat((tar_pad_mask, line['tar_pad_mask']), dim=0)
            tar_pred_mask  = torch.cat((tar_pred_mask, line['tar_pred_mask']), dim=0)
            song_name.append(line['song_name'])
            start.append(line['start'])
            total.append(line['total'])
    return {
        'z_src': z_src,
        'z_tar' : z_tar,
        'src_pad_mask': src_pad_mask,
        'tar_pad_mask': tar_pad_mask,
        'tar_pred_mask': tar_pred_mask,
        'song_name': song_name,
        'start': start,
        'total': total
    }

def decouple_line(line):
    # Slice z to z_src and z_tar and make the corresponding masks
    z = line['z'] # (1, 2 * z_len)
    pred_mask = line['pred_mask']
    sep_mask = line['sep_mask']
    pad_mask = line['pad_mask']
    song_name = line['song_name']
    start = line['start']
    total = line['total']
    
    z_len = int(z.shape[1] / 2)
    sep_idx = torch.argmax(sep_mask, dim=1)
    
    # z_src and src_mask
    z_src = z[:, :sep_idx]
    src_pad_mask = torch.zeros_like(z_src)
    # Pad
    z_src = torch.nn.functional.pad(z_src, (0, z_len - z_src.shape[1]))
    src_pad_mask = torch.nn.functional.pad(src_pad_mask, (0, z_len - src_pad_mask.shape[1]), value=1)

    # z_tar and tar_mask
    z_tar = z[:, sep_idx+1: sep_idx+1+z_len] # Exclude the sep token
    tar_pad_mask = pad_mask[:, sep_idx+1: sep_idx+1+z_len]
    tar_pred_mask = 1 - tar_pad_mask

    return {
        'z_src': z_src,
        'z_tar' : z_tar,
        'src_pad_mask': src_pad_mask,
        'tar_pad_mask': tar_pad_mask,
        'tar_pred_mask': tar_pred_mask,
        'song_name': song_name,
        'start': start,
        'total': total
    }
        
                
def build_z2z_loader(src_dir, tar_dir, batch_size, shuffle=True, random_offset=True, controlnet=False):
    dataset = Z2ZDataset(src_dir, tar_dir, random_offset, controlnet=False)
    if not controlnet:
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=mr_collate, shuffle=shuffle)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ctrl_collate, shuffle=shuffle)
    print(f'Dataset {src_dir} - {tar_dir}')
    print(f'Size {len(dataset)}')
    return loader
    
if __name__ == '__main__':
    loader = build_z2z_loader(uglobals.MUSDB18_TEST_VOCALS_Z_DIR, uglobals.MUSDB18_TEST_ACC_Z_DIR, 1, random_offset=False, shuffle=False)
    for batch in loader:
        print(batch['z'][0, :10])
        print(batch['z'][0, -10:])
        exit()