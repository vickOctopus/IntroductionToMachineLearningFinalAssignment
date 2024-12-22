# 实验报告素材

# 第一部分：数据集说明

## Fashion-MNIST 数据集简介

Fashion-MNIST 是一个替代 MNIST 手写数字数据集的图像数据集。它包含了 10 个类别的服装图像：

- 0: T-shirt/top（T恤/上衣）
- 1: Trouser（裤子）
- 2: Pullover（套衫）
- 3: Dress（连衣裙）
- 4: Coat（外套）
- 5: Sandal（凉鞋）
- 6: Shirt（衬衫）
- 7: Sneaker（运动鞋）
- 8: Bag（包）
- 9: Ankle boot（短靴）

### 数据集特点
- 图像大小：28x28 像素
- 颜色：灰度图像（单通道）
- 训练集：60,000 张图片
- 测试集：10,000 张图片
- 标签：10 个类别（0-9）

## 数据集划分

本项目对原始数据集进行了如下划分：

1. 训练集（80%）
   - 数量：48,000 张图片
   - 用途：模型训练
   - 比例：原始训练集的 80%

2. 验证集（10%）
   - 数量：6,000 张图片
   - 用途：早停和模型选择
   - 比例：原始训练集的 10%

3. 测试集（独立）
   - 数量：10,000 张图片
   - 用途：最终性能评估
   - 来源：原始测试集

---

# 第二部分：性能指标说明

## 模型效率指标

1. 参数量 (Parameters)
   - 计算公式：`total_params = sum(p.numel() for p in model.parameters())`
   - 物理含义：模型中可训练参数的总数，反映模型复杂度
   - 单位：个

2. 推理速度 (Inference Speed)
   - 计算公式：`samples_per_second = total_samples / total_time`
   - 物理含义：每秒可以处理的样本数量，反映模型推理效率
   - 单位：samples/s

3. 平均推理时间 (Avg Inference Time)
   - 计算公式：`avg_time = total_time / total_samples * 1000`
   - 物理含义：处理单个样本的平均时间
   - 单位：ms/sample

### 训练效率指标

1. 训练时长 (Training Time)
   - 计算公式：`total_train_time = sum(epoch_times)`
   - 物理含义：完成所有训练轮次所需的总时间
   - 单位：秒(s)

2. 收敛轮数 (Convergence Epoch)
   - 计算公式：当损失变化小于阈值(`threshold=0.001`)时的轮次
   - 物理含义：模型达到稳定状态所需的训练轮数
   - 单位：轮(epoch)

### 性能评估指标

1. 准确率 (Accuracy)
   - 计算公式：`accuracy = correct_predictions / total_samples`
   - 物理含义：模型预测正确的样本比例
   - 范围：[0, 1]

2. 损失值 (Loss)
   - 计算公式：交叉熵损失(CrossEntropyLoss)
   - 物理含义：预测结果与真实标签的差异程度
   - 范围：[0, ∞)

---

# 第三部分：神经网络参数实验

## 实验结果文件位置

1. 原始实验数据
   - 路径：`experiment_logs/experiment_results.json`
   - 内容：包含所有参数组合的详细实验结果

2. 性能分析报告
   - 路径：`experiment_logs/performance_analysis.md`
   - 内容：不同参数配置的性能对比分析

3. 复杂度分析报告
   - 路径：`experiment_logs/complexity_analysis.md`
   - 内容：不同网络结构的复杂度和效率分析

4. loss曲线可视化结果
   - 路径：`visualization_results/loss_curves.png`
   - 内容：训练过程中的损失曲线变化



