#!/bin/bash
#SBATCH --partition=nvidia_gpu        # 指定分区
#SBATCH --ntasks=1                    # 请求1个任务
#SBATCH --gres=gpu:1                  # 请求6个GPU
#SBATCH --cpus-per-task=18           # 请求每个任务48个CPU核
#SBATCH --nodelist=gpu1              # 指定运行节点为 gpu2

# 在 gpu2 节点上执行训练脚本
python train.py
