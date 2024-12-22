#!/bin/bash

# 指定 Python 版本
PYTHON_VERSION=python3.9

# 创建虚拟环境
$PYTHON_VERSION -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

echo "虚拟环境已创建并安装依赖"

# 创建必要的目录
mkdir -p data/raw models experiment_logs visualization_results