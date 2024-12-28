import torch
import torch.nn as nn
import torch.optim as optim
from .fashion_mnist import train_loader, test_loader
import json
from datetime import datetime
import os
import time
import numpy as np

# 更新日志目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
log_dir = os.path.join(project_root, 'experiment_logs')

# 检测可用设备
if torch.backends.mps.is_available():  # Mac的Metal加速
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

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

# 实验配置 
EXPERIMENTS = {
    'learning_rates': [0.01, 0.001],  # 两种学习率
    'optimizers': {
        'Adam': optim.Adam  # 保持使用Adam优化器
    },
    'activations': ['relu', 'leakyrelu'],  
    'conv_layers': [2, 3, 4]  # 三种络深度
}

def calculate_model_complexity(model):
    """计算模型复杂度"""
    # 简化为只返回参数总量
    return {
        'total_params': sum(p.numel() for p in model.parameters())
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
            if i >= 50:  # 只测试50个批次
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
    }

def calculate_training_efficiency(epoch_times):
    """计算训练效率指标"""
    return {
        'total_train_time': sum(epoch_times),
        'avg_epoch_time': np.mean(epoch_times)
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

def train_and_evaluate(model, optimizer, lr=0.001, epochs=5):
    """训练和评估模型"""
    start_time = time.time()
    epoch_times = []
    
    # 将模型移到GPU
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    
    # 初始化早停机制，增加patience
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)  # 从2增加到3
    
    # 训练
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据移��GPU
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
        
        # 早停检查
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break
    
    # 计算训练效率
    training_efficiency = calculate_training_efficiency(epoch_times)
    
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
                    print(f"实验完成: {model_name}, 准确率: {result['accuracy']:.4f}")
                except Exception as e:
                    print(f"实验失败: {model_name}")
                    print(f"错误信息: {e}")
    
    # 保存实验结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 进行分析
    from .analyze import create_performance_table, save_results_table, create_complexity_analysis
    df_perf = create_performance_table(results)
    save_results_table(df_perf)
    
    # 生成复杂度分析
    df_complexity = create_complexity_analysis(results)
    
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