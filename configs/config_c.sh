NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
--data=datasets/cifar10/cifar10-train.zip --cond=0 --arch=karras --dtype=bf16 --attn-resolutions=8,4 \
--uncertainty=True --log_wandb=False