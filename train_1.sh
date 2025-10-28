#!/bin/bash
#SBATCH -J mode0
#SBATCH -p p-RTX2080
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
python train.py --name mode0 --dataroot /mntnfs/med_data5/wanyue/real-vs-fake/results/LNP --classes=face --arch=res50 --mode=0

