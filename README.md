# Fashion-MNIST 神经网络参数实验

## 项目说明

本项目是宁波大学机器学习导论课程的期末作业，实现了一个基于 Fashion-MNIST 数据集的神经网络实验平台，用于研究不同网络参数对模型性能的影响。

### 主要功能
- 基础CNN模型训练与评估
- 自动化的网络参数实验
- 详细的性能分析报告
- 可视化结果展示

## 快速开始

### 环境要求
- Windows 10/11
- Python 3.9
  - 如果没有安装Python，脚本会自动安装
  - 或访问 [Python官网](https://www.python.org/downloads/) 手动安装

### 使用步骤

1. 下载项目
   - 下载并解压项目文件

2. 运行安装脚本
   ```bash
   # 右键点击文件夹空白处，选择"在终端中打开"
   #输入
   scripts\setup.bat
   ```
   安装脚本会：
   - 检查并安装Python（如果需要）
   - 创建虚拟环境
   - 安装所需依赖
   - 创建必要的目录

3. 运行实验
   ```bash
   # 在项目文件夹中打开终端
   python run.py
   ```

### 可能的问题和解决方案

1. Python安装失败
   - 访问 [Python官网](https://www.python.org/downloads/)
   - 下载并安装Python 3.9
   - 安装时勾选"Add Python to PATH"

2. Python版本不正确
   - 检查Python版本：`python --version`
   - 如果版本不是3.9：
     - 卸载当前Python版本
     - 从 [Python 3.9下载页面](https://www.python.org/downloads/release/python-3913/) 下载3.9版本
     - 安装时勾选"Add Python to PATH"
   - 如果同时安装了多个Python版本：
     - Win+R 输入 `sysdm.cpl`
     - 高级 -> 环境变量
     - 在Path中调整Python 3.9的优先级

3. 依赖安装失败
   - 检查网络连接
   - 尝试使用管理员权限运行终端
   - 如果PyTorch安装失败，访问 [PyTorch官网](https://pytorch.org/get-started/locally/)

4. 权限问题
   - 右键点击终端，选择"以管理员身份运行"
   - 重新运行安装脚本

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