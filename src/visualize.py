# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os

# 更新路径计算
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
viz_dir = os.path.join(project_root, 'visualization_results')

# 确保可视化结果目录存在
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)

def plot_loss_curves(train_losses, val_losses, test_losses=None, save_path=None):
    """绘制训练过程的损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制训练集损失
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    # 绘制验证集损失
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    
    # 绘制测试集损失
    if test_losses is not None:
        plt.plot(epochs, test_losses, 'g-.', label='Test Loss')
    
    plt.title('Training, Validation and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    if save_path is None:
        save_path = os.path.join(viz_dir, 'training_curves.png')
    plt.savefig(save_path)
    plt.close() 