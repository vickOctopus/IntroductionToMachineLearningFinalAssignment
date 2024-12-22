import torch
import torch.nn as nn
import torch.optim as optim
from .fashion_mnist import train_loader, test_loader
import json
from datetime import datetime
import os

# 更新日志目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
log_dir = os.path.join(project_root, 'experiment_logs')

def log_training(log_file, message):
    """记录训练信息到文件"""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 1. 基础CNN（作为基准模型）
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 深层CNN
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.classifier(x)
        return x

def train_and_evaluate(model, lr=0.001, epochs=5, log_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练配置
    if log_file:
        log_training(log_file, f"\n训练配置:")
        log_training(log_file, f"学习率: {lr}")
        log_training(log_file, f"训练轮数: {epochs}")
        log_training(log_file, f"设备: {device}")
    
    # 训练
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            if i % 100 == 99:
                message = f'[Epoch {epoch + 1}] Loss: {running_loss / 100:.3f}'
                print(message)
                if log_file:
                    log_training(log_file, message)
                running_loss = 0.0
        
        # 记录每个epoch的平均损失
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        if log_file:
            log_training(log_file, f'Epoch {epoch + 1} 平均损失: {avg_epoch_loss:.3f}')
    
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
    if log_file:
        log_training(log_file, f'\n最终测试准确率: {accuracy:.4f}')
    
    return {
        'accuracy': accuracy,
        'epoch_losses': epoch_losses
    }

def experiment():
    # 创建实验记录文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    results_file = os.path.join(log_dir, f'results_{timestamp}.json')
    
    # 记录实验开始
    log_training(log_file, f"实验开始时间: {datetime.now()}")
    
    models = {
        'BaseCNN': BaseCNN(),
        'DeepCNN': DeepCNN()
    }
    learning_rates = [0.001, 0.0001]
    results = {}
    
    for model_name, model in models.items():
        log_training(log_file, f"\n\n测试模型: {model_name}")
        model_results = {}
        
        for lr in learning_rates:
            log_training(log_file, f"\n学习率: {lr}")
            result = train_and_evaluate(model, lr=lr, log_file=log_file)
            model_results[str(lr)] = result
            print(f"准确率: {result['accuracy']:.4f}")
        
        results[model_name] = model_results
    
    # 保存实验结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 记录实验结束
    log_training(log_file, f"\n\n实验结束时间: {datetime.now()}")
    
    return results, log_file, results_file

if __name__ == '__main__':
    results, log_file, results_file = experiment()
    print(f"\n实验日志已保存到: {log_file}")
    print(f"实验结果已保存到: {results_file}")
    
    # 打印最终结果摘要
    print("\n最终结果摘要:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for lr, result in model_results.items():
            print(f"  学习率 {lr}: {result['accuracy']:.4f}") 