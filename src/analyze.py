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
    """生成性能对比表格 - 分析不同参数对性能的影响"""
    records = []
    for model_name, result in results.items():
        try:
            parts = model_name.split('_')
            if len(parts) != 4:
                continue
                
            _, activation, conv_layers, lr = parts
            record = {
                '激活函数': activation,
                '网络层数': int(conv_layers),
                '学习率': float(lr),
                '准确率': result['accuracy'],
                '训练时长(秒)': result['efficiency']['training_efficiency']['total_train_time'],
                '平均每轮训练时间(秒)': result['efficiency']['training_efficiency']['avg_epoch_time']
            }
            records.append(record)
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    # 先进行分析计算
    analysis = {
        '激活函数分析': df.groupby('激活函数')['准确率'].agg(['mean', 'max', 'min']),
        '网络层数分析': df.groupby('网络层数')['准确率'].agg(['mean', 'max', 'min']),
        '学习率分析': df.groupby('学习率')['准确率'].agg(['mean', 'max', 'min'])
    }
    
    # 格式化分析结果
    for key in analysis:
        analysis[key] = analysis[key].round(4)
    
    # 然后格式化原始数据
    df['准确率'] = df['准确率'].apply(lambda x: f"{x:.4f}")
    df['训练时长(秒)'] = df['训练时长(秒)'].apply(lambda x: f"{x:.2f}")
    df['平均每轮训练时间(秒)'] = df['平均每轮训练时间(秒)'].apply(lambda x: f"{x:.2f}")
    df['学习率'] = df['学习率'].apply(lambda x: f"{x:.4f}")
    
    return df, analysis

def save_analysis_results(perf_df, perf_analysis, complexity_df):
    """保存分析结果"""
    # 保存性能分析结果
    perf_csv = os.path.join(log_dir, 'performance_analysis.csv')
    perf_df.to_csv(perf_csv, index=False)
    
    # 保存性能分析的Markdown报告
    perf_md = os.path.join(log_dir, 'performance_analysis.md')
    with open(perf_md, 'w', encoding='utf-8') as f:
        f.write("# 模型性能分析\n\n")
        f.write("## 原始数据\n")
        f.write(perf_df.to_markdown(index=False))
        f.write("\n\n## 参数影响分析\n")
        for name, analysis in perf_analysis.items():
            f.write(f"\n### {name}\n")
            f.write(analysis.to_markdown())
            f.write("\n")
    
    # 保存复杂度分析结果
    complexity_csv = os.path.join(log_dir, 'complexity_analysis.csv')
    complexity_df.to_csv(complexity_csv, index=False)
    
    # 保存复杂度分析的Markdown报告
    complexity_md = os.path.join(log_dir, 'complexity_analysis.md')
    with open(complexity_md, 'w', encoding='utf-8') as f:
        f.write("# 模型复杂度分析\n\n")
        f.write(complexity_df.to_markdown(index=False))

def create_complexity_analysis(results):
    """生成模型复杂度分析表格 - 关注计算资源消耗"""
    records = []
    for model_name, result in results.items():
        try:
            parts = model_name.split('_')
            if len(parts) != 4:
                continue
                
            _, activation, conv_layers, lr = parts
            
            efficiency = result['efficiency']
            training_eff = efficiency['training_efficiency']
            
            # 计算单个样本的平均测试时长
            total_samples = 10000  # test_loader的总样本数
            inference_time = total_samples / efficiency['inference_speed']
            avg_sample_time = inference_time / total_samples
            
            record = {
                '激活函数': activation,
                '网络层数': int(conv_layers),
                '学习率': float(lr),
                '参数量': int(efficiency['params_count']),
                '模型大小(MB)': int(efficiency['params_count']) * 4 / (1024 * 1024),
                '训练总时长(秒)': float(training_eff['total_train_time']),
                '单样本测试时长(毫秒)': avg_sample_time * 1000,
                '推理速度(样本/秒)': float(efficiency['inference_speed'])
            }
            records.append(record)
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    # 格式化数值
    df['参数量'] = df['参数量'].apply(lambda x: f"{x:,}")
    df['模型大小(MB)'] = df['模型大小(MB)'].apply(lambda x: f"{x:.2f}")
    df['训练总时长(秒)'] = df['训练总时长(秒)'].apply(lambda x: f"{x:.2f}")
    df['单样本测试时长(毫秒)'] = df['单样本测试时长(毫秒)'].apply(lambda x: f"{x:.3f}")
    df['推理速度(样本/秒)'] = df['推理速度(样本/秒)'].apply(lambda x: f"{x:.2f}")
    
    return df

def main():
    # 使用固定的结果文件
    results_file = os.path.join(log_dir, 'experiment_results.json')
    
    # 加载结果
    results = load_experiment_results(results_file)
    
    # 创建性能表格
    perf_df, perf_analysis = create_performance_table(results)
    
    # 保存结果
    save_analysis_results(perf_df, perf_analysis, create_complexity_analysis(results))
    
    print(f"Results tables have been saved to:")
    print(f"CSV: {perf_csv}")
    print(f"Markdown: {perf_md}")
    print("\nTop 5 Configurations:")
    print(perf_df.head().to_markdown())

if __name__ == "__main__":
    main() 