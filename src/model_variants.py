import torch
import torch.nn as nn
import torch.optim as optim
from .fashion_mnist import train_loader, test_loader
import json
from datetime import datetime
import os
from .visualize import plot_loss_curves
import time
import numpy as np

# 更新日志目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
log_dir = os.path.join(project_root, 'experiment_logs')

# 检测并使用MPS（用于Mac）或CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

def log_training(log_file, message):
    """记录训练信息到文件"""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 基础CNN（可配置参数）
class BaseCNN(nn.Module):
    def __init__(self, 
                 activation='relu',
                 conv_layers=2,
                 channels=[32, 64],
                 kernel_size=3):
        super(BaseCNN, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建卷积层
        layers = []
        in_channels = 1
        for i in range(conv_layers):
            out_channels = channels[i] if i < len(channels) else channels[-1]
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                self.activation,
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # 计算全连接层输入维度
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.features(x)
            fc_input = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 512),
            self.activation,
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 实验配置 - 增加参数组合
EXPERIMENTS = {
    'learning_rates': [0.01, 0.001],  # 两种学习率
    'optimizers': {
        'Adam': optim.Adam  # 保持使用Adam优化器
    },
    'activations': ['relu', 'leakyrelu'],  # 修改这里：去掉下划线
    'conv_layers': [2, 3, 4]  # 三种��络深度
}

def calculate_model_complexity(model):
    """计算模型复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def measure_inference_time(model, test_loader, num_batches=50):
    """测量推理时间"""
    device = next(model.parameters()).device
    total_time = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            total_time += end_time - start_time
            total_samples += batch_size
    
    avg_time_per_sample = total_time / total_samples
    return {
        'total_time': total_time,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': total_samples / total_time
    }

def measure_memory_usage(model, test_loader):
    """测量内存使用"""
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    # 运行一个批次来测量内存
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _ = model(inputs)
    
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        max_memory = 0  # CPU模式下不测量内存
        
    return {
        'max_memory_mb': max_memory
    }

def evaluate_model_efficiency(model, test_loader):
    """评估模型效率"""
    # 1. 计算模型复杂度
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. 测量推理时间
    device = next(model.parameters()).device
    total_time = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 50:  # 修改这里：从100改为50个批次
                break
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            total_time += end_time - start_time
            total_samples += batch_size
    
    return {
        'params_count': total_params,                          # 模型参数量
        'inference_speed': total_samples / total_time,         # 推理速度（样本/秒）
        'avg_inference_time': total_time / total_samples       # 平均推理时间（秒/样本）
    }

def calculate_training_efficiency(epoch_times, losses):
    """计算训练效率指标"""
    # 收敛判定：当损失变化小于阈值时认为收敛
    def find_convergence_epoch(losses, threshold=0.001):
        for i in range(1, len(losses)):
            if abs(losses[i] - losses[i-1]) < threshold:
                return i
        return len(losses)
    
    return {
        'total_train_time': sum(epoch_times),                 # 总训练时间
        'convergence_epoch': find_convergence_epoch(losses),  # 收敛轮次
        'avg_epoch_time': np.mean(epoch_times)                # 平均每轮训练时间
    }

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_and_evaluate(model, optimizer, lr=0.001, epochs=3):
    """训练和评估模型"""
    start_time = time.time()
    epoch_times = []
    epoch_losses = []
    
    # 将模型移到GPU
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)  # 使用更激进���早停策略
    
    # 训练
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据移到GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(avg_epoch_loss)
        
        # 早停检查
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break
    
    # 计算训练效率
    training_efficiency = calculate_training_efficiency(epoch_times, epoch_losses)
    
    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    
    # 评估模型效率
    efficiency_metrics = evaluate_model_efficiency(model, test_loader)
    efficiency_metrics.update({
        'training_efficiency': training_efficiency
    })
    
    return {
        'accuracy': accuracy,
        'efficiency': efficiency_metrics
    }

def experiment():
    """进行实验"""
    results_file = os.path.join(log_dir, 'experiment_results.json')
    results = {}
    
    total_experiments = (len(EXPERIMENTS['activations']) * 
                        len(EXPERIMENTS['conv_layers']) * 
                        len(EXPERIMENTS['learning_rates']))
    current_experiment = 0
    
    print(f"\n总共需要进行 {total_experiments} 组实验")
    print("实验配置:")
    print(f"激活函数: {EXPERIMENTS['activations']}")
    print(f"网络层数: {EXPERIMENTS['conv_layers']}")
    print(f"学习率: {EXPERIMENTS['learning_rates']}")
    
    for activation in EXPERIMENTS['activations']:
        for conv_layers in EXPERIMENTS['conv_layers']:
            for lr in EXPERIMENTS['learning_rates']:
                current_experiment += 1
                print(f"\n实验进度: [{current_experiment}/{total_experiments}]")
                print(f"配置: activation={activation}, layers={conv_layers}, lr={lr}")
                
                try:
                    model_name = f"BaseCNN_{activation}_{conv_layers}_{lr}"
                    model = BaseCNN(activation=activation, 
                                  conv_layers=conv_layers,
                                  channels=[32, 64],
                                  kernel_size=3)
                    result = train_and_evaluate(model, 
                                              EXPERIMENTS['optimizers']['Adam'], 
                                              lr=lr)
                    results[model_name] = result
                    print(f"实验完��: {model_name}, 准确率: {result['accuracy']:.4f}")
                except Exception as e:
                    print(f"实验失败: {model_name}")
                    print(f"错误信息: {e}")
    
    # 保存实验结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 直接进行分析
    from .analyze import create_performance_table, save_results_table
    df = create_performance_table(results)
    save_results_table(df)
    
    print("\n性能分析果:")
    print("\nTop 5 Configurations:")
    print(df.head().to_markdown())
    
    return results, results_file

if __name__ == '__main__':
    results, results_file = experiment()
    print(f"\n实验结果已保存到: {results_file}")
    
    # 打印最终结果摘要
    print("\n最终结果摘要:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for lr, result in model_results.items():
            print(f"  学习率 {lr}: {result['accuracy']:.4f}") 