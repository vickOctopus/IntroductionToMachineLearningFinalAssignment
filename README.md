# Fashion-MNIST 神经网络参数实验

## 项目说明

本项目是宁波大学机器学习导论课程的期末作业，实现了一个基于 Fashion-MNIST 数据集的神经网络实验平台，用于研究不同网络参数对模型性能的影响。

### 主要功能
- 基础CNN模型训练与评估
- 自动化的网络参数实验
- 详细的性能分析报告
- 可视化结果展示

### 实验内容
1. 基础模型训练
   - 使用默认参数训练基准CNN模型
   - 保存训练过程和结果

2. 参数实验
   - 测试不同的网络配置
   - 包括激活函数、层数、学习率等

3. 性能分析
   - 生成详细的性能报告
   - 包括参数量、训练时间、准确率等

## 快速开始

### 详细安装步骤

#### 1. 安装 Python

##### Windows:
1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载 Python 3.9.x 安装包
3. 运行安装程序，勾选"Add Python to PATH"
4. 验证安装：
```bash
python --version  # 应显示 Python 3.9.x
```

##### macOS:
```bash
# 使用 Homebrew 安装
brew install python@3.9

# 验证安装
python3.9 --version
```

#### 2. 运行环境设置脚本

##### Windows:
```bash
# 在项目文件夹中打开终端（或命令提示符）
# 右键点击文件夹空白处，选择"在终端中打开"

# 运行设置脚本（会自动创建虚拟环境并安装依赖）
scripts\setup.bat
```

##### macOS/Linux:
```bash
# 在项目文件夹中打开终端
# 右键点击文件夹，选择"新建位于文件夹位置的终端窗口"

# 添加执行权限
chmod +x scripts/setup.sh

# 运行设置脚本
./scripts/setup.sh
```

#### 3. 下载数据集

##### 自动下载（推荐）：
可以先进行步骤4，程序会自动下载 Fashion-MNIST 数据集

##### 手动下载（可选）：
如果自动下载失败，可以手动下载并放置数据集：

1. 下载链接：
   - GitHub: [Fashion-MNIST官方数据集](https://github.com/zalandoresearch/fashion-mnist#get-the-data)

2. 需要下载的文件：
   - train-images-idx3-ubyte.gz
   - train-labels-idx1-ubyte.gz
   - t10k-images-idx3-ubyte.gz
   - t10k-labels-idx1-ubyte.gz

3. 解压文件并放置到 `data/FashionMNIST/raw/` 目录下

#### 4. 运行实验

##### Windows:
```bash
# 在项目文件夹中打开终端
python run.py
```

##### macOS/Linux:
```bash
# 在项目文件夹中打开终端
python3 run.py
# 或
./run.py  # 如果添加了执行权限
```

## 项目结构

```
project_root/
├── data/                    # 数据集目录
│   └── FashionMNIST/       # Fashion-MNIST数据集
│       └── raw/            # 原始数据文件
├── experiment_logs/         # 实验结果和分析
│   ├── experiment_results.json     # 实验原始数据
│   ├── performance_analysis.md     # 性能分析报告
│   └── complexity_analysis.md      # 复杂度分析报告
├── models/                  # 保存的模型文件
│   ├── fashion_mnist_model.pth     # 训练好的模型
│   └── training_history.npz        # 训练历史数据
├── scripts/                 # 环境设置脚本
│   ├── setup.bat                   # Windows环境设置
│   └── setup.sh                    # Linux/Mac环境设置
├── src/                     # 源代码目录
│   ├── __init__.py                 # 包初始化文件
│   ├── fashion_mnist.py            # 基础模型实现
│   ├── parameter_experiments.py    # 参数实验实现
│   ├── analyze.py                  # 结果分析工具
│   └── visualize.py                # 可视化工具
├── visualization_results/   # 可视化结果
│   └── loss_curves.png            # 损失曲线图
├── requirements.txt         # 项目依赖
├── README.md               # 项目说明
└── run.py                  # 主运行脚本
```

### 文件说明

#### 核心代码
- `src/fashion_mnist.py`: 实现基础CNN模型和训练流程
- `src/parameter_experiments.py`: 实现不同参数组合的实验
- `src/analyze.py`: 实现实验结果的分析和报告生成
- `src/visualize.py`: 实现训练过程的可视化

#### 配置文件
- `requirements.txt`: 列出项目所需的Python包
- `scripts/setup.bat`和`setup.sh`: 环境配置脚本

#### 运行脚本
- `run.py`: 主程序入口，执行训练和实验

#### 输出目录
- `experiment_logs/`: 存放实验结果和分析报告
- `models/`: 存放训练好的模型
- `visualization_results/`: 存放可视化结果

### 输出文件说明

#### 实验结果
- `experiment_results.json`: 包含所有实验配置的原始结果
- `performance_analysis.md`: 性能对比分析报告
- `complexity_analysis.md`: 模型复杂度分析报告

#### 模型文件
- `fashion_mnist_model.pth`: 保存的模型参数
- `training_history.npz`: 训练过程中的损失和准确率数据

#### 可视化结果
- `loss_curves.png`: 训练、验证和测试损失曲线