#!/bin/bash
#SBATCH --partition=hygon_dcu        # 指定分区
#SBATCH --ntasks=1                    # 请求1个任务
#SBATCH --gres=dcu:1                  # 请求6个GPU
#SBATCH --cpus-per-task=18           # 请求每个任务48个CPU核
#SBATCH --nodelist=dcu1              # 指定运行节点为 gpu2


# # 激活conda环境
# conda activate dcu  # 假设你的环境名是 dcu

# 加载必要的模块
module load compiler/dtk/23.04
module load mpi/openmpi/4.1.1/gcc-9.3.0  # 加载适配的MPI模块
module load compiler/gcc/9.3.0



# 在 gpu2 节点上执行训练脚本
python train.py
