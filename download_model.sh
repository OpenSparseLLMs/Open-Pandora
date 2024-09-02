#!/bin/bash

# 检查是否提供了参数
if [ "$#" -lt 1 ]; then
  echo "使用方式: $0 <model-name> [repo-type]"
  exit 1
fi

MODEL_NAME="$1"
REPO_TYPE="${2:-models}"  # 设置默认值为 "models"

# 提取模型名称中的最后一部分，使用/作为分隔符
MODEL_FOLDER=$(basename "$MODEL_NAME")

# 执行替换model-name的命令
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --repo-type ${REPO_TYPE%?} ${MODEL_NAME} --local-dir /mnt/petrelfs/share_data/quxiaoye/"$REPO_TYPE"/"$MODEL_FOLDER" --local-dir-use-symlinks False