#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragGPT-GA专业学术图表生成器
===========================
生成符合学术论文标准的高质量模型亮点展示图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# 设置专业的学术风格
plt.style.use('default')
# 检查可用的样式
try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')  # 备用样式

# 设置字体为 Times New Roman（学术标准）
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def create_professional_fraggpt_highlights():
    """创建专业的FragGPT-GA亮点展示图"""
    
    # 创建主图布局
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1, 1])
    
    # 主标题
    fig.suptitle('FragGPT-GA: A Novel Hybrid Framework for Molecular Optimization', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 子图1: 性能对比雷达图 (左上，占两个位置)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # 使用真实实验数据
    methods = ['RGA', 'AutoGrow4.0', 'MARS', 'GEGL', 'REINVENT', 'FragGPT-GA']
    
    # 标准化数据 (转换为0-1范围便于比较)
    docking_scores = [12.869, 12.474, 9.257, 10.450, 12.010, 13.458]  # 取绝对值
    qed_scores = [0.742, 0.748, 0.709, 0.643, 0.445, 0.764]
    sa_scores = [2.473, 2.497, 2.450, 2.990, 2.596, 2.014]  # 越小越好，需要反转
    novelty = [100, 100, 100, 100, 100, 100]  # 新颖性百分比
    
    # 标准化处理
    docking_norm = [(x - min(docking_scores)) / (max(docking_scores) - min(docking_scores)) for x in docking_scores]
    qed_norm = [(x - min(qed_scores)) / (max(qed_scores) - min(qed_scores)) for x in qed_scores]
    sa_norm = [(max(sa_scores) - x) / (max(sa_scores) - min(sa_scores)) for x in sa_scores]  # 反转
    novelty_norm = [x/100 for x in novelty]
    
    # 创建雷达图数据
    metrics = ['Docking Score', 'Drug-likeness (QED)', 'Synthetic Accessibility', 'Novelty']
    
    # 选择代表性方法进行对比
    selected_methods = ['RGA', 'AutoGrow4.0', 'MARS', 'FragGPT-GA']
    selected_indices = [0, 1, 2, 5]
    
    # 准备雷达图数据
    values = []
    for i in selected_indices:
        values.append([docking_norm[i], qed_norm[i], sa_norm[i], novelty_norm[i]])
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 清除ax1并重新设置为极坐标
    ax1.clear()
    ax1 = fig.add_subplot(gs[0, :2], projection='polar')
    
    # 颜色方案
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f', '#ff7fff']
    
    for i, (method, color) in enumerate(zip(selected_methods, colors)):
        values_plot = values[i] + values[i][:1]  # 闭合
        if method == 'FragGPT-GA':
            ax1.plot(angles, values_plot, 'o-', linewidth=3, label=method, color=color)
            ax1.fill(angles, values_plot, alpha=0.25, color=color)
        else:
            ax1.plot(angles, values_plot, 'o-', linewidth=2, label=method, color=color, alpha=0.7)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title('(A) Multi-dimensional Performance Comparison', pad=20, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # 子图2: 动态掩码策略 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    
    generations = np.arange(1, 26)
    n_initial = 3
    n_final = 1
    G_max = 25
    
    n_mask = n_initial + (generations - 1) / (G_max - 1) * (n_final - n_initial)
    
    ax2.plot(generations, n_mask, 'b-', linewidth=3, marker='o', markersize=4, 
             markerfacecolor='orange', markeredgecolor='blue', markeredgewidth=1)
    ax2.fill_between(generations, n_mask, alpha=0.3, color='lightblue')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Masked Fragments')
    ax2.set_title('(B) Dynamic Masking Strategy', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加公式
    ax2.text(0.5, 0.95, r'$n_{mask}(g) = n_{initial} + \frac{g-1}{G_{max}-1}(n_{final} - n_{initial})$', 
             transform=ax2.transAxes, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # 子图3: 性能柱状图对比 (左下)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 选择TOP-1 docking scores进行对比
    top1_methods = ['RGA', 'AutoGrow4.0', 'MARS', 'GEGL', 'REINVENT', 'FragGPT-GA']
    top1_scores = [-12.869, -12.474, -9.257, -10.450, -12.010, -13.458]
    top1_errors = [0.473, 0.839, 0.791, 1.040, 0.833, 0.442]
    
    # 创建颜色，FragGPT-GA用特殊颜色突出
    bar_colors = ['lightblue'] * 5 + ['darkgreen']
    
    bars = ax3.bar(range(len(top1_methods)), np.abs(top1_scores), 
                   yerr=top1_errors, capsize=3, color=bar_colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # 突出显示最佳性能
    bars[-1].set_color('#2E8B57')
    bars[-1].set_edgecolor('darkgreen')
    bars[-1].set_linewidth(2)
    
    ax3.set_xlabel('Methods')
    ax3.set_ylabel('Binding Affinity (|kcal/mol|)')
    ax3.set_title('(C) TOP-1 Docking Score Comparison', fontweight='bold')
    ax3.set_xticks(range(len(top1_methods)))
    ax3.set_xticklabels(top1_methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, score, err) in enumerate(zip(bars, top1_scores, top1_errors)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + err + 0.1,
                 f'{abs(score):.2f}', ha='center', va='bottom', 
                 fontweight='bold' if i == len(bars)-1 else 'normal',
                 fontsize=9)
    
    # 子图4: QED vs SA散点图 (中下)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 实际QED和SA数据
    qed_data = [0.742, 0.748, 0.709, 0.643, 0.445, 0.764]
    sa_data = [2.473, 2.497, 2.450, 2.990, 2.596, 2.014]
    method_names = ['RGA', 'AutoGrow4.0', 'MARS', 'GEGL', 'REINVENT', 'FragGPT-GA']
    
    # 绘制散点图
    colors_scatter = ['red', 'orange', 'blue', 'purple', 'brown', 'green']
    sizes = [80] * 5 + [120]  # FragGPT-GA用更大的点
    
    for i, (qed, sa, method, color, size) in enumerate(zip(qed_data, sa_data, method_names, colors_scatter, sizes)):
        marker = 's' if method == 'FragGPT-GA' else 'o'
        ax4.scatter(qed, sa, c=color, s=size, marker=marker, 
                   edgecolors='black', linewidth=1.5, alpha=0.8, label=method)
    
    ax4.set_xlabel('Drug-likeness (QED)')
    ax4.set_ylabel('Synthetic Accessibility (SA)')
    ax4.set_title('(D) Drug-likeness vs Synthetic Accessibility', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加理想区域
    ax4.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax4.text(0.72, 2.3, 'Ideal Region\n(High QED, Low SA)', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
             fontsize=8)
    
    # 子图5: GPT优势示意图 (右下)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 绘制GPT优势
    advantages = [
        'Population Distribution\nExpansion',
        'Learned Chemical\nKnowledge Integration',
        'Accelerated\nConvergence',
        'Scaffold Hopping\nCapability'
    ]
    
    # 创建圆形布局
    center_x, center_y = 0.5, 0.5
    radius = 0.3
    
    # 中心GPT圆圈
    center_circle = plt.Circle((center_x, center_y), 0.15, 
                              facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax5.add_patch(center_circle)
    ax5.text(center_x, center_y, 'Fragment\nGPT', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # 四个优势
    angles_adv = np.linspace(0, 2*np.pi, 4, endpoint=False)
    for i, (angle, advantage) in enumerate(zip(angles_adv, advantages)):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # 优势框
        bbox = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1, 
                              boxstyle="round,pad=0.01", 
                              facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5)
        ax5.add_patch(bbox)
        ax5.text(x, y, advantage, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 连接线
        dx = x - center_x
        dy = y - center_y
        line_start_x = center_x + 0.15 * dx / radius
        line_start_y = center_y + 0.15 * dy / radius
        ax5.plot([line_start_x, x], [line_start_y, y], 'b-', linewidth=2, alpha=0.7)
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('(E) GPT Core Advantages', fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 生成专业图表
    fig = create_professional_fraggpt_highlights()
    
    # 保存到论文目录
    output_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/fraggpt_ga_professional.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    print(f"专业FragGPT-GA展示图已保存至: {output_path}")
    
    # 也保存为EPS格式（学术论文标准）
    eps_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/fraggpt_ga_professional.eps"
    fig.savefig(eps_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='eps')
    print(f"EPS格式已保存至: {eps_path}")
    
    plt.show()
