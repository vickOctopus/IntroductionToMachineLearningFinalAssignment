#!/bin/bash

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 创建必要的目录
mkdir -p data/raw models experiment_logs visualization_results