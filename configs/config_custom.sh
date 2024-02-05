NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
--data=datasets/cifar10/cifar10-train.zip --cond=0 --arch=karras --dtype=bf16 --attn-resolutions=8,4 \
--uncertainty=True --log-wandb=True --t-ref 70000 --lr 0.01 --batch 4096 --augment=0 \
--channel-mult=1,2,3,4 --num-blocks=3 --num-channels=128 \
--ref-path=fid_refs/cifar10-all.npz --use-dataset-stat=True