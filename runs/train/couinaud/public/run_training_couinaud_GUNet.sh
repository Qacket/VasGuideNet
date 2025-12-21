#!/bin/bash
dataset=public
model_name=GUNet
# with_contrast_loss  True
# replace_strategy=cats

for fold in 1 2 3 4 5; do

  echo "Running fold $fold"

  # 设置临时目录
  export TMPDIR=/data0/scj/tmp
  mkdir -p $TMPDIR

  # 创建日志目录
  logdir="/data0/scj/mixed_train/${dataset}/Couinaud_${model_name}_${fold}"
  mkdir -p "$logdir"

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$((23456 + fold)) \
    /data0/scj/code_final/main_lits.py \
    --distributed \
    --logdir "$logdir" \
    --val_every 100 \
    --batch_size 2 \
    --save_intervals 10 \
    --model_name "$model_name" \
    --workers 6 \
    --num_class 9 \
    --optim_lr 1e-4 \
    --max_epochs 2000 \
    --sw_batch_size 4 \
    --train_dir "/data0/scj/datasets/肝八段数据集/Couinaud_lmdb/Couinaud_public/fold${fold}/train/" \
    --val_dir "/data0/scj/datasets/肝八段数据集/Couinaud_lmdb/Couinaud_public/fold${fold}/val/"  \
    2>&1 | tee "${logdir}/train.log"
done
