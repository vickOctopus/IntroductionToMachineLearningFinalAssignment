# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from .fashion_mnist import CNN, classes  # 使用相对导入

# 更新路径计算
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
root_path = os.path.join(project_root, 'data')
viz_dir = os.path.join(project_root, 'visualization_results')
models_dir = os.path.join(project_root, 'models')

def plot_training_process(train_losses, train_accs, val_losses, val_accs):
    """绘制训练过程中的损失和准确率变化"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(model, test_loader, classes):
    """绘制混淆矩阵"""
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加类别标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(model, test_loader, classes, num_samples=10):
    """展示一些样本预测结果"""
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}',
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_predictions.png'))
    plt.close()

if __name__ == '__main__':
    # 加载训练好的模型
    model = CNN()
    try:
        model.load_state_dict(torch.load(os.path.join(models_dir, 'fashion_mnist_model.pth'), weights_only=True))
        print("模型加载成功")
    except:
        print("错误：未找到模型文件 fashion_mnist_model.pth")
        exit(1)
    
    # 加载训练历史数据
    try:
        history = np.load(os.path.join(models_dir, 'training_history.npz'))
        train_losses = history['train_losses']
        train_accs = history['train_accs']
        val_losses = history['val_losses']
        val_accs = history['val_accs']
        print("训练历史数据加载成功")
    except:
        print("错误：未找到训练历史数据 training_history.npz")
        exit(1)
    
    # 生成可视化
    plot_training_process(train_losses, train_accs, val_losses, val_accs)
    print("已生成训练过程曲线图")
    
    # 创建测试数据加载器，使用正确的数据路径
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root_path,  # 使用正确的数据路径
        train=False,
        download=False,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    plot_confusion_matrix(model, test_loader, classes)
    print("已生成混淆矩阵图")
    
    plot_sample_predictions(model, test_loader, classes)
    print("已生成样本预测结果图") 