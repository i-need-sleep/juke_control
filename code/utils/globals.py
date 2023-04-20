# Data
MUSDB18_PATH = '../data/musdb18'
MUSDB18_PROCESSED_PATH = f'{MUSDB18_PATH}/processed'

DEBUG_STEM_PATH = f'{MUSDB18_PATH}/train/A Classic Education - NightOwl.stem.mp4'
DEBUG_VOCAL_PATH = f'{MUSDB18_PATH}/debug/A Classic Education - NightOwl_vocals.wav'
DEBUG_ACC_PATH = f'{MUSDB18_PATH}/debug/A Classic Education - NightOwl_acc.wav'

DEBUG_VOCAL_Z_PATH = f'{MUSDB18_PATH}/debug/debug_z_vocals.pt'
DEBUG_ACC_Z_PATH = f'{MUSDB18_PATH}/debug/debug_z_acc.pt'

# Output
OUT_PATH = '../results/outputs'
MUSDB18_ORACLE = f'{OUT_PATH}/musdb18/oracle'
MUSDB18_TRAIN_VOCALS_Z_DIR = f'{MUSDB18_ORACLE}/train/vocals/z'
MUSDB18_TRAIN_ACC_Z_DIR = f'{MUSDB18_ORACLE}/train/acc/z'
MUSDB18_TEST_VOCALS_Z_DIR = f'{MUSDB18_ORACLE}/test/vocals/z'
MUSDB18_TEST_ACC_Z_DIR = f'{MUSDB18_ORACLE}/test/acc/z'
MUSDB18_Z_OUT = f'{OUT_PATH}/musdb18/z_out'
MUSDB18_WAV_OUT = f'{OUT_PATH}/musdb18/wav_out'