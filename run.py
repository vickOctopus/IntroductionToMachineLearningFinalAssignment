#!/usr/bin/env python3
import sys
import os
from pathlib import Path

def setup_path():
    """设置Python路径"""
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    # 将src目录添加到Python路径
    sys.path.insert(0, str(project_root))

def ensure_venv():
    """确保在虚拟环境中运行"""
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / 'venv'
    
    # 检查虚拟环境是否存在
    if not venv_path.exists():
        print("错误: 未找到虚拟环境")
        print("请先运行 setup.sh (Linux/macOS) 或 setup.bat (Windows) 创建虚拟环境")
        sys.exit(1)
    
    # 检查是否已在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix
    if not in_venv:
        # 确定操作系统类型
        if sys.platform == 'win32':
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = venv_path / 'bin' / 'python3'
        
        if not python_path.exists():
            print("错误: 虚拟环境��乎已损坏，请重新运行 setup.sh 或 setup.bat")
            sys.exit(1)
        
        print("切换到虚拟环境...")
        # 使用虚拟环境的Python重新执行当前脚本
        os.execl(str(python_path), str(python_path), *sys.argv)

def print_usage():
    """打印使用说明"""
    print("\n使用方法:")
    print("  python run.py [mode]")
    print("\n可用模式:")
    print("  train      - 训练基础模型 (默认模式)")
    print("  variants   - 训练模型变体并进行实验")
    print("  visualize  - 可视化训练结果")
    print("\n示例:")
    print("  python run.py train     # 训练基础模型")
    print("  python run.py variants  # 进行模型变体实验")
    print("  python run.py visualize # 可视化结果")

def main():
    """主函数"""
    # 确保在虚拟环境中运行
    ensure_venv()
    
    # 设置路径
    setup_path()
    
    try:
        # 解析命令行参数
        mode = 'train'  # 默认模式
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
        
        if mode == 'train':
            # 训练基础模型
            from src.fashion_mnist import train
            print("\n开始训练基础模型...")
            train_losses, train_accs, val_losses, val_accs = train(epochs=5)
            print("\n训练完成！")
            print("模型和训练历史已保存到 models/ 目录")
            
        elif mode == 'variants':
            # 训练模型变体
            from src.model_variants import experiment
            print("\n开始模型变体实验...")
            results, log_file, results_file = experiment()
            print(f"\n实验日志已保存到: {log_file}")
            print(f"实验结果已保存到: {results_file}")
            
        elif mode == 'visualize':
            # 可视化结果
            import src.visualize
            print("\n开始生成可视化结果...")
            
        else:
            print(f"\n错误: 未知的模式 '{mode}'")
            print_usage()
            sys.exit(1)
            
    except ImportError as e:
        print(f"\n错误: 导入模块失败 ({e})")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 