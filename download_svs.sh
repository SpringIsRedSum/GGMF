#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "用法: $0 <manifest文件> <保存目录>"
    exit 1
fi

MANIFEST=$1
SAVE_DIR=$2

# 检查manifest文件是否存在
if [ ! -f "$MANIFEST" ]; then
    echo "错误: manifest文件 '$MANIFEST' 不存在"
    exit 1
fi

# 创建保存目录(如果不存在)
mkdir -p "$SAVE_DIR"

# 读取manifest文件并下载SVS文件
while IFS=$'\t' read -r id filename md5 size; do
    if [[ $filename == *.svs ]]; then
        url="https://api.gdc.cancer.gov/data/$id"
        output_file="$SAVE_DIR/$filename"
        
        echo "正在下载: $filename"
        wget -c -O "$output_file" "$url" 2>&1 | \
        stdbuf -o0 awk '/[.] +[0-9][0-9]?%/ { print substr($0,63,3) }' | \
        stdbuf -o0 tr '\n' '\r' | \
        sed -u 's/.*/下载进度: &%/'
        
        echo -e "\n下载完成: $filename"
    fi
done < "$MANIFEST"

echo "所有SVS文件下载完成"
