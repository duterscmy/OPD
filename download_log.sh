#!/bin/bash

# 检查是否提供了文件名参数
if [ $# -eq 0 ]; then
    echo "错误: 请指定要复制的文件名"
    echo "用法: $0 <文件名>"
    echo "示例: $0 checkpoints_math500_num_generation8_block32_t0.6_lr5e-6_only_rollout_rank.log"
    exit 1
fi

# 获取文件名参数
FILENAME="$1"

# 远程服务器配置
REMOTE_HOST="u6nc.aip2.isambard"
REMOTE_PATH=""
LOCAL_PATH="logs"

# 创建本地目录（如果不存在）
mkdir -p "$LOCAL_PATH"

# 执行复制
echo "正在复制 $FILENAME 从 $REMOTE_HOST 到 $LOCAL_PATH ..."
scp "${REMOTE_HOST}:${FILENAME}" "${LOCAL_PATH}/"

# 检查复制结果
if [ $? -eq 0 ]; then
    echo "✓ 复制成功完成！"
    echo "文件已保存到: $LOCAL_PATH/$FILENAME"
else
    echo "✗ 复制失败，请检查网络连接和文件路径"
    exit 1
fi
