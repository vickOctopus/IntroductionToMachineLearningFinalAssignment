# Fashion MNIST 图像分类项目

这个项目实现了一个基于CNN的Fashion MNIST图像分类系统。

## 项目结构 

project_root/
├── data/ # 数据目录
│ └── FashionMNIST/
│ └── raw/ # 原始数据文件
├── models/ # 模型和训练历史数据
├── fashion_mnist/ # 代码目录
│ ├── fashion_mnist.py # 主训练脚本
│ ├── model_variants.py # 模型变体实验
│ └── visualize.py # 可视化工具
├── experiment_logs/ # 实验日志目录
└── visualization_results/ # 可视化结果目录

## 主要脚本说明

### 1. fashion_mnist.py

基础训练脚本，运行后会：
- 加载和预处理Fashion MNIST数据集
- 训练基础CNN模型
- 在测试集上评估模型性能
- 生成以下文件：
  - `models/fashion_mnist_model.pth`：训练好的模型
  - `models/training_history.npz`：训练过程的历史数据

### 2. model_variants.py

模型变体实验脚本，运行后会：
- 测试不同的CNN架构（BaseCNN和DeepCNN）
- 使用不同的学习率进行训练
- 生成以下文件：
  - `experiment_logs/experiment_YYYYMMDD_HHMMSS.log`：详细的训练日志
  - `experiment_logs/results_YYYYMMDD_HHMMSS.json`：实验结果数据

### 3. visualize.py

可视化脚本，运行后会：
- 加载训练好的模型和历史数据
- 生成以下文件：
  - `visualization_results/training_curves.png`：训练过程中的损失和准确率曲线
  - `visualization_results/confusion_matrix.png`：混淆矩阵图
  - `visualization_results/sample_predictions.png`：样本预测结果图


## 运行顺序建议

1. 首先运行 `fashion_mnist.py` 训练基础模型
2. 然后可以运行 `visualize.py` 查看训练结果的可视化
3. 如果需要实验不同的模型架构，运行 `model_variants.py`

## 注意事项

1. 运行前确保 `data/FashionMNIST/raw/` 目录下有所需的数据文件
2. `visualize.py` 需要在运行 `fashion_mnist.py` 之后才能使用
3. 每次运行 `model_variants.py` 都会生成新的实验日志文件
4. 所有生成的文件会自动保存在相应目录中

## 依赖库

- PyTorch
- NumPy
- Matplotlib
- torchvision