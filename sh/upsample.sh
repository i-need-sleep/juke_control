#!/bin/bash
#SBATCH --job-name=updample       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=256GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/eval/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/eval/%x%A.err            # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p gpu                   # 有GPU的partition
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/juke_control/code/models/jukebox   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate jukebox          # 调用 virtual env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export HF_HOME=/l/users/yichen.huang/misc/cache
export XDG_CACHE_HOME=/l/users/yichen.huang/misc/cache
python jukebox/sample.py \
    --model=1b_lyrics \
    --name=upsample_ctrl_vocal2acc \
    --levels=3 \
    --mode=upsample \
    --codes_file=../../../results/outputs/urmp/z_out/controlnet_midi2audio_lr_decay3e-4_checkpoint_step_6001.pth.tar/21_Rejouissance_cl_tbn_tba_8003_25887.pt \
    --sample_length_in_seconds=16 \
    --total_sample_length_in_seconds=180 \
    --sr=44100 \
    --n_samples=1 \
    --hop_fraction=0.5,0.5,0.125
echo "FINISH"                       # 输出起始信息
