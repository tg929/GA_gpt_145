#!/usr/bin/env python3
"""
绘制三个模型（AutoGrow4.0, RGA, FragGPT-GA）在10种蛋白质上对接分数的小提琴图对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def extract_docking_scores(base_dir, model_name, score_column_index):
    """
    从.smi文件中提取对接分数
    
    Args:
        base_dir: 模型数据目录
        model_name: 模型名称
        score_column_index: 对接分数所在的列索引
    
    Returns:
        DataFrame: 包含蛋白质名称、模型名称和对接分数的数据框
    """
    all_scores = []
    protein_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    protein_dirs = [d for d in protein_dirs if d not in ['__pycache__']]  # 过滤掉非蛋白质目录
    
    print(f"Processing {model_name}, found proteins: {sorted(protein_dirs)}")
    
    for protein in sorted(protein_dirs):
        protein_dir = os.path.join(base_dir, protein)
        smi_file = os.path.join(protein_dir, f"{protein}.smi")
        
        if os.path.exists(smi_file):
            try:
                with open(smi_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t')
                            if len(parts) > score_column_index:
                                try:
                                    score = float(parts[score_column_index])
                                    all_scores.append({
                                        'Protein': protein,
                                        'Model': model_name,
                                        'Docking_Score': score
                                    })
                                except ValueError:
                                    continue
                print(f"  {protein}: {len([s for s in all_scores if s['Protein'] == protein and s['Model'] == model_name])} scores")
            except Exception as e:
                print(f"Error reading {smi_file}: {e}")
        else:
            print(f"Warning: {smi_file} not found")
    
    return pd.DataFrame(all_scores)

def main():
    # 数据目录
    base_path = "/data1/ytg/medium_models/GA_gpt/compare_baselins_drawing"
    
    # 提取三个模型的数据
    print("Extracting AutoGrow4.0 data...")
    autogrow_data = extract_docking_scores(
        os.path.join(base_path, "autogrow4.0"), 
        "AutoGrow4.0", 
        4  # 第5列（索引4）
    )
    
    print("\nExtracting RGA data...")
    rga_data = extract_docking_scores(
        os.path.join(base_path, "rga"), 
        "RGA", 
        2  # 第3列（索引2）
    )
    
    print("\nExtracting FragGPT-GA data...")
    fraggpt_data = extract_docking_scores(
        os.path.join(base_path, "FragGPT_GA"), 
        "FragGPT-GA", 
        1  # 第2列（索引1）
    )
    
    # 合并所有数据
    all_data = pd.concat([autogrow_data, rga_data, fraggpt_data], ignore_index=True)
    
    # 检查数据
    print(f"\nTotal data points: {len(all_data)}")
    print("Data summary by model:")
    print(all_data.groupby('Model')['Docking_Score'].agg(['count', 'mean', 'std', 'min', 'max']))
    
    print("\nData summary by protein:")
    print(all_data.groupby('Protein')['Docking_Score'].agg(['count', 'mean', 'std']))
    
    # 获取所有蛋白质列表并排序
    proteins = sorted(all_data['Protein'].unique())
    print(f"\nProteins: {proteins}")
    
    # 设置绘图风格
    plt.style.use('default')
    
    # 创建子图 - 2行5列布局显示10种蛋白质
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()  # 展平为一维数组便于索引
    
    # 定义颜色（参考你提供的小提琴图配色）
    colors = ['#82C182', '#FF9999', '#87CEEB']  # 绿色、粉色、蓝色
    model_order = ['AutoGrow4.0', 'RGA', 'FragGPT-GA']
    
    # 为每种蛋白质绘制子图
    for idx, protein in enumerate(proteins):
        ax = axes[idx]
        
        # 准备该蛋白质的数据
        protein_data = all_data[all_data['Protein'] == protein]
        
        # 为每个模型准备数据
        violin_data = []
        for model in model_order:
            model_protein_data = protein_data[protein_data['Model'] == model]['Docking_Score'].values
            violin_data.append(model_protein_data)
            print(f"{protein} - {model}: {len(model_protein_data)} data points")
        
        # 绘制小提琴图
        violin_parts = ax.violinplot(violin_data, positions=range(1, len(model_order) + 1), 
                                   showmeans=True, showmedians=True, widths=0.7)
        
        # 设置颜色
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # 设置其他元素的颜色
        violin_parts['cmeans'].set_colors('red')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_colors('blue')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cbars'].set_colors('black')
        violin_parts['cmaxes'].set_colors('black')
        violin_parts['cmins'].set_colors('black')
        
        # 添加散点图显示数据分布
        for i, model in enumerate(model_order):
            model_data = violin_data[i]
            if len(model_data) > 0:
                # 添加少量随机抖动以避免重叠
                x_pos = np.random.normal(i + 1, 0.02, len(model_data))
                ax.scatter(x_pos, model_data, alpha=0.4, s=15, color=colors[i], 
                          edgecolors='black', linewidth=0.3)
        
        # 设置子图属性
        ax.set_xticks(range(1, len(model_order) + 1))
        ax.set_xticklabels(['AutoGrow4.0', 'RGA', 'FragGPT-GA'], fontsize=8, rotation=0)
        ax.set_title(f'{protein.upper()}', fontsize=12, fontweight='bold', pad=10)
        
        # 设置Y轴
        if len(protein_data) > 0:
            y_min = protein_data['Docking_Score'].min() - 0.5
            y_max = protein_data['Docking_Score'].max() + 0.5
            ax.set_ylim(y_min, y_max)
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # 只在左边列显示Y轴标签
        if idx % 5 == 0:
            ax.set_ylabel('Docking Score (kcal/mol)', fontsize=10)
        
        # 添加TOP1统计信息（最佳分数，即最小值）
        stats_text = []
        for i, model in enumerate(model_order):
            model_data = protein_data[protein_data['Model'] == model]['Docking_Score']
            if len(model_data) > 0:
                top1_score = model_data.min()  # 对接分数越小越好，所以用min()
                stats_text.append(f"{model[:4]}: {top1_score:.1f}")
        
        # 在右上角添加TOP1统计信息
        if stats_text:
            ax.text(0.98, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 设置总标题
    fig.suptitle('Docking Score Comparison Across Three Models for 10 Protein Targets', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 创建图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                   edgecolor='black', label=model_order[i]) 
                      for i in range(len(model_order))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=3, fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # 保存图片
    output_path = "/data1/ytg/medium_models/GA_gpt/papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/violin_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nViolin plot saved to: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 输出每种蛋白质的详细统计
    print("\nDetailed Statistics by Protein:")
    for protein in proteins:
        print(f"\n=== {protein.upper()} ===")
        protein_data = all_data[all_data['Protein'] == protein]
        for model in model_order:
            model_data = protein_data[protein_data['Model'] == model]['Docking_Score']
            if len(model_data) > 0:
                print(f"{model}:")
                print(f"  Count: {len(model_data)}")
                print(f"  Mean: {model_data.mean():.2f}")
                print(f"  Std: {model_data.std():.2f}")
                print(f"  Min: {model_data.min():.2f}")
                print(f"  Max: {model_data.max():.2f}")
                print(f"  Median: {model_data.median():.2f}")
            else:
                print(f"{model}: No data")

if __name__ == "__main__":
    main()
