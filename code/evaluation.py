import os

import numpy as np
from scipy.io import wavfile

import mir_eval
import aubio
import audiolazy

from utils.eval_utils import average, prec_recall_f1

def eval_sdr(dir):
    # Since the slices may have varying size, eval for each slice and take the macro average
    pred_sdrs = []
    oracle_sdrs = []
    for song_idx, song_name in enumerate(os.listdir(dir)):
        pred_path = f'{dir}/{song_name}/pred/item_0.wav'
        oracle_path = f'{dir}/{song_name}/oracle/item_0.wav'
        tar_path = f'{dir}/{song_name}/tar/item_0.wav'
        
        pred = wavfile.read(pred_path)[1].reshape(1, -1)
        oracle = wavfile.read(oracle_path)[1].reshape(1, -1)
        tar = wavfile.read(tar_path)[1].reshape(1, -1)

        # Skip all-zero reference sources
        if np.sum(tar) == 0:
            continue

        pred_sdr, _, _, _ = mir_eval.separation.bss_eval_sources(reference_sources=tar, estimated_sources=pred)
        oracle_sdr, _, _, _ = mir_eval.separation.bss_eval_sources(reference_sources=tar, estimated_sources=oracle)

        pred_sdrs.append(float(pred_sdr))
        oracle_sdrs.append(float(oracle_sdr))
    
    pred_avg_sdr = sum(pred_sdrs) / len(pred_sdrs)
    oracle_avg_sdr = sum(oracle_sdrs) / len(oracle_sdrs)

    print(f'# Samples: {len(pred_sdrs)}')
    print(f'Prediction: Average SDR: {pred_avg_sdr}')
    print(f'Oracle: Average SDR: {oracle_avg_sdr}')
    return

def eval_beats(dir):
    pred_scores = []
    oracle_scores = []
    for song_idx, song_name in enumerate(os.listdir(dir)):
        pred_path = f'{dir}/{song_name}/pred/item_0.wav'
        oracle_path = f'{dir}/{song_name}/oracle/item_0.wav'
        tar_path = f'{dir}/{song_name}/tar/item_0.wav'
        
        pred_beats = aubio_beats(pred_path)
        oracle_beats = aubio_beats(oracle_path)
        tar_beats = aubio_beats(tar_path)
        
        pred_score = mir_eval.beat.f_measure(reference_beats=tar_beats, estimated_beats=pred_beats)
        oracle_score = mir_eval.beat.f_measure(reference_beats=tar_beats, estimated_beats=oracle_beats)

        pred_scores.append(pred_score)
        oracle_scores.append(oracle_score)
    
    pred_avg_score = sum(pred_scores) / len(pred_scores)
    oracle_avg_score = sum(oracle_scores) / len(oracle_scores)

    print(f'# Samples: {len(pred_scores)}')
    print(f'Prediction: Average F1: {pred_avg_score}')
    print(f'Oracle: Average F1: {oracle_avg_score}')
    return

def aubio_beats(path, hop_s=512):
    s = aubio.source(path, hop_size=hop_s)
    samplerate = s.samplerate
    o = aubio.tempo(samplerate=samplerate, hop_size=hop_s)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
        total_frames += read
        if read < hop_s:
            break
    return np.array(beats)

def eval_pitches(dir):
    # Bag-of-pitches evaluation against the source pitches
    pred_precs, pred_racalls, pred_f1s = [], [], []
    oracle_precs, oracle_recalls, oracle_f1s = [], [], []
    tar_precs, tar_recalls, tar_f1s = [], [], []
    for song_idx, song_name in enumerate(os.listdir(dir)):
        pred_path = f'{dir}/{song_name}/pred/item_0.wav'
        oracle_path = f'{dir}/{song_name}/oracle/item_0.wav'
        src_path = f'{dir}/{song_name}/src/item_0.wav'
        tar_path = f'{dir}/{song_name}/tar/item_0.wav'
        
        pred_pits = aubio_pitches(pred_path)
        oracle_pits = aubio_pitches(oracle_path)
        src_pits = aubio_pitches(src_path)
        tar_pits = aubio_pitches(tar_path)
        
        pred_prec, pred_racall, pred_f1 = prec_recall_f1(pred_pits, src_pits)
        oracle_prec, oracle_recall, oracle_f1 = prec_recall_f1(oracle_pits, src_pits)
        tar_prec, tar_recall, tar_f1 = prec_recall_f1(tar_pits, src_pits)

        pred_precs.append(pred_prec)
        pred_racalls.append(pred_racall)
        pred_f1s.append(pred_f1)

        oracle_precs.append(oracle_prec)
        oracle_recalls.append(oracle_recall)
        oracle_f1s.append(oracle_f1)

        tar_precs.append(tar_prec)
        tar_recalls.append(tar_recall)
        tar_f1s.append(tar_f1)

    print(f'# Samples: {len(pred_precs)}')
    print(f'Prediction: Macro prec/recall/f1: {average(pred_precs)}, {average(pred_racalls)}, {average(pred_f1s)}')
    print(f'Oracle: Macro prec/recall/f1: {average(oracle_precs)}, {average(oracle_recalls)}, {average(oracle_f1s)}')
    print(f'Target: Macro prec/recall/f1: {average(tar_precs)}, {average(tar_recalls)}, {average(tar_f1s)}')
    return

def aubio_pitches(path, hop_s=512, thresh=0.5):
    s = aubio.source(path, hop_size=hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = aubio.pitch("yin", hop_size=hop_s, samplerate=samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        if confidence > thresh:
            # print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
            pitches += [pitch]
            confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    # Round pitches to midi pitches
    out = []
    for pitch in pitches:
        if pitch > 0:
            out.append(round(audiolazy.lazy_midi.freq2midi(pitch)) % 12)
    out = list(set(out))
    return out

if __name__ == '__main__':
    # eval_sdr('../results/outputs/musdb18/wav_out/finetune_srcsep_checkpoint_step_17001.pth.tar')
    eval_pitches('../results/outputs/musdb18/wav_out/finetune_srcsep_checkpoint_step_17001.pth.tar')