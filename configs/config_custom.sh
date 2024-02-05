NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
--data=datasets/cifar10/cifar10-train.zip --cond=0 --arch=karras --dtype=bf16 \
--uncertainty=True --log_wandb=True --t-ref 70000 --lr 0.01 --batch 2048 --num-channels=128 --augment=0