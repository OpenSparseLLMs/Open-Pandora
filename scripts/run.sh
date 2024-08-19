# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64
echo $CUDA_VISIBLE_DEVICES
# eval $activate_cuda118
# source activate dynamicrafter
HOST_GPU_NUM=4
# args
config_file=scripts/config.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="output/ckp_align"

mkdir -p $save_root/$name

## run
python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=10042 --node_rank=0 \
trainer_stage1.py \
--model_path /mnt/petrelfs/tianjie/projects/WorldModel/Pandora/models/Pandora-7B \
--base $config_file \
--train \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1

## debugging
# python trainer_stage1.py  --model_path /mnt/petrelfs/tianjie/projects/WorldModel/Pandora/models/Pandora-7B  --base $config_file --train --logdir output/ckp --devices 1
