import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import sys

# 在导入部分添加错误处理
try:
    from src.visualize import plot_loss_curves
    print("成功导入 plot_loss_curves 函数")
except ImportError as e:
    print(f"导入 plot_loss_curves 失败: {e}")
    print(f"Python 路径: {sys.path}")
    raise

# 更新路径计算
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
data_path = os.path.join(project_root, 'data', 'FashionMNIST', 'raw')
root_path = os.path.join(project_root, 'data')
models_dir = os.path.join(project_root, 'models')

print(f"当前目录: {current_dir}")
print(f"数据目录: {data_path}")
print(f"根目录: {root_path}")

# 检查必需的文件是否存在
required_files = [
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte'
]

print("\n检查数据集...")
if not all(os.path.exists(os.path.join(data_path, f)) for f in required_files):
    print("数据集不存在，开始下载...")
else:
    print("数据集已存在，跳过下载")

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.FashionMNIST(
    root=root_path,
    train=True, 
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root=root_path,
    train=False, 
    download=True,
    transform=transform
)

# 修改数据加载部分
train_size = int(0.8 * len(train_dataset))  # 使用80%的数据进行训练
val_size = int(0.1 * len(train_dataset))    # 使用10%的数据作为验证集
train_dataset, val_dataset, _ = torch.utils.data.random_split(
    train_dataset, 
    [train_size, val_size, len(train_dataset) - train_size - val_size],
    generator=torch.Generator().manual_seed(42)  # 设置随机种子确保可重复性
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=128,
    shuffle=True,
    pin_memory=True,  # 加快数据传输到GPU的度
    num_workers=4     # 使用多个工作进程加载数据
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

# 定义类别名称
classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print("数据集加载完成！")
print(f"训练集大小: {len(train_dataset)} (原始数据的80%)")
print(f"验证集大小: {len(val_dataset)} (原始数据的10%)")
print(f"测试集大小: {len(test_dataset)}")

# 在文件开头添加设备检测
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 一个卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 全连层
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = CNN().to(device)  # 将模型移到设备
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model structure:")
print(model)

# 在test函数前添加evaluate函数
def evaluate(model, data_loader):
    """评估模型在给定数据集上的性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据移到设备
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

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

def train(epochs=5):
    """训练函数"""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        batch_count = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到设备
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0
        
        # 每个epoch结束后收集数据
        train_losses.append(epoch_loss / batch_count)
        train_accs.append(correct / total)
        
        # 验证集评估
        val_loss, val_acc = evaluate(model, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 测试集评估
        test_loss, test_acc = evaluate(model, test_loader)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch + 1} - '
              f'Train Loss: {train_losses[-1]:.3f}, '
              f'Val Loss: {val_loss:.3f}, '
              f'Test Loss: {test_loss:.3f}, '
              f'Train Acc: {train_accs[-1]:.3f}, '
              f'Val Acc: {val_acc:.3f}')
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break
    
    # 保存训练结果
    save_results(model, train_losses, train_accs, val_losses, val_accs, test_losses)
    
    # 在训练结束后自动生成可视化
    try:
        print("开始生成损失曲线...")
        print(f"train_losses 长度: {len(train_losses)}")
        print(f"val_losses 长度: {len(val_losses)}")
        print(f"test_losses 长度: {len(test_losses)}")
        plot_loss_curves(train_losses, val_losses, test_losses)
        print("损失曲线生成完成")
    except Exception as e:
        print(f"生成损失曲线时出错: {e}")
        raise
    
    return train_losses, train_accs, val_losses, val_accs, test_losses

# 在train函数后添加test函数
def test():
    """测试函数：评估模型在测试集上的性能"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'\nOverall test accuracy: {100 * correct / total:.2f}%')
    
    # 打印每个类别的准确率
    for i in range(10):
        print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

# 在文件开头添加模型保存路径
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 修改保存部分的代码
def save_results(model, train_losses, train_accs, val_losses, val_accs, test_losses):
    """保存模型和训练结果"""
    # 确保模型保存目录存在
    os.makedirs(models_dir, exist_ok=True)
    
    # 构建保存路径
    model_path = os.path.join(models_dir, 'fashion_mnist_model.pth')
    history_path = os.path.join(models_dir, 'training_history.npz')
    
    # 保存模型
    try:
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 保存训练历史
    try:
        np.savez(history_path,
            train_losses=train_losses,
            train_accs=train_accs,
            val_losses=val_losses,
            val_accs=val_accs,
            test_losses=test_losses
        )
        print(f"训练历史已保存到: {history_path}")
    except Exception as e:
        print(f"保存训练历史时出错: {e}")

# 修改主函数部分
if __name__ == '__main__':
    print("\nStarting training...")
    train_losses, train_accs, val_losses, val_accs, test_losses = train(epochs=5)

    print("\nStarting testing...")
    test()

    # 保存结果
    save_results(model, train_losses, train_accs, val_losses, val_accs, test_losses)