NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
--data=datasets/cifar10/cifar10-train.zip --cond=0 --arch=adm --dtype=bf16 --attn-resolutions=8,4 \
--uncertainty=True --log-wandb=False --num-channels=192 --augment=0 \
--ref-path=fid_refs/cifar10-all.npz