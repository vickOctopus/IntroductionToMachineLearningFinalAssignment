import pandas as pd
import json
from pathlib import Path
import os

# 更新路径计算
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
viz_dir = os.path.join(project_root, 'visualization_results')
log_dir = os.path.join(project_root, 'experiment_logs')  # 用于存放实验结果和性能分析

# 确保可视化结果目录存在
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)

def load_experiment_results(results_file):
    """加载实验结果"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_performance_table(results):
    """生成性能对比表格"""
    records = []
    print(f"\n处理的实验结果数量: {len(results)}")
    print("模型配置列表:")
    
    for model_name, result in results.items():
        try:
            print(f"正在处理模型: {model_name}")  # 打印每个模型名称
            parts = model_name.split('_')
            print(f"拆分结果: {parts}")  # 打印拆分结果
            
            if len(parts) == 4:
                _, activation, conv_layers, lr = parts
                record = {
                    'Activation': activation,
                    'Conv Layers': conv_layers,
                    'Learning Rate': lr,
                    'Parameters': result['efficiency']['params_count'],
                    'Inference Speed (samples/s)': result['efficiency']['inference_speed'],
                    'Training Time (s)': result['efficiency']['training_efficiency']['total_train_time'],
                    'Accuracy': result['accuracy']
                }
                records.append(record)
                print(f"成功处理模型: {model_name}")
            else:
                print(f"警告: 模型名称格式不正确: {model_name}, 部分数量: {len(parts)}")
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 排序和格式化
    df = df.sort_values('Accuracy', ascending=False)
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x:.4f}")
    df['Inference Speed (samples/s)'] = df['Inference Speed (samples/s)'].apply(lambda x: f"{x:.2f}")
    df['Training Time (s)'] = df['Training Time (s)'].apply(lambda x: f"{x:.2f}")
    
    return df

def save_results_table(df):
    """保存结果表格到experiment_logs目录"""
    # 使用"性能分析"作为文件名
    csv_path = os.path.join(log_dir, 'performance_analysis.csv')
    df.to_csv(csv_path, index=False)
    
    # 同时保存为Markdown格式以便查看
    md_path = os.path.join(log_dir, 'performance_analysis.md')
    with open(md_path, 'w') as f:
        f.write("# Performance Analysis\n\n")
        f.write(df.to_markdown())
    
    return csv_path, md_path

def create_complexity_analysis(results):
    """Generate model complexity analysis table"""
    records = []
    for model_name, result in results.items():
        try:
            _, activation, conv_layers, lr = model_name.split('_')
            
            efficiency = result['efficiency']
            training_eff = efficiency['training_efficiency']
            
            record = {
                'Model Config': f"{activation}-{conv_layers}layers-lr{lr}",
                'Parameters': int(efficiency['params_count']),
                'Inference Speed (samples/s)': float(efficiency['inference_speed']),
                'Training Time (s)': float(training_eff['total_train_time']),
                'Accuracy': float(result['accuracy'])
            }
            records.append(record)
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    # Create DataFrame and sort
    df = pd.DataFrame(records)
    df = df.sort_values('Accuracy', ascending=False)
    
    # 格式化数值列
    df['Parameters'] = df['Parameters'].apply(lambda x: f"{x:,}")
    df['Inference Speed (samples/s)'] = df['Inference Speed (samples/s)'].apply(lambda x: f"{x:.2f}")
    df['Training Time (s)'] = df['Training Time (s)'].apply(lambda x: f"{x:.2f}")
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x:.4f}")
    
    # Save results
    csv_path = os.path.join(log_dir, 'complexity_analysis.csv')
    df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(log_dir, 'complexity_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Model Complexity Analysis\n\n")
        f.write(df.to_markdown(index=False))
    
    print(f"\nComplexity analysis saved to:")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return df

def main():
    # 使用固定的结果文件
    results_file = os.path.join(log_dir, 'experiment_results.json')
    
    # 加载结果
    results = load_experiment_results(results_file)
    
    # 创建性能表格
    df = create_performance_table(results)
    
    # 保存结果
    csv_path, md_path = save_results_table(df)
    
    print(f"Results tables have been saved to:")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print("\nTop 5 Configurations:")
    print(df.head().to_markdown())

if __name__ == "__main__":
    main() 