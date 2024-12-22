#!/usr/bin/env python3
import sys
import os
from pathlib import Path

def setup_path():
    """设置Python路径"""
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))

def ensure_venv():
    """确保在虚拟环境中运行"""
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / 'venv'
    
    if not venv_path.exists():
        print("错误: 未找到虚拟环境")
        print("请先运行 scripts/setup.sh (Linux/macOS) 或 scripts/setup.bat (Windows) 创建虚拟环境")
        sys.exit(1)
    
    in_venv = hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix
    if not in_venv:
        if sys.platform == 'win32':
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = venv_path / 'bin' / 'python3'
        
        if not python_path.exists():
            print("错误: 虚拟环境似乎已损坏，请重新运行 scripts/setup.sh 或 scripts/setup.bat")
            sys.exit(1)
        
        print("切换到虚拟环境...")
        os.execl(str(python_path), str(python_path), *sys.argv)

def main():
    """主函数"""
    ensure_venv()
    setup_path()
    
    try:
        # 1. 先训练基础模型
        from src.fashion_mnist import train
        print("\n第一步: 训练基础模型...")
        train_losses, train_accs, val_losses, val_accs, test_losses = train(epochs=5)
        print("\n基础模型训练完成！")
        
        # 2. 然后进行参数实验
        from src.parameter_experiments import experiment
        print("\n第二步: 开始网络参数实验...")
        results, results_file = experiment()
        print(f"\n实验结果已保存到: {results_file}")
        
        print("\n所有步骤完成！")
            
    except ImportError as e:
        print(f"\n错误: 导入模块失败 ({e})")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 