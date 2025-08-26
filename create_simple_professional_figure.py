#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragGPT-GA专业学术图表生成器（简化版）
===================================
生成符合学术论文标准的高质量模型亮点展示图
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# 设置专业的学术风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_fraggpt_professional_figure():
    """创建专业的FragGPT-GA学术展示图"""
    
    # 创建2x2布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('FragGPT-GA: Hybrid Framework Performance and Innovation', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # === 图1: 性能对比柱状图 (使用真实数据) ===
    ax1.set_title('(A) Docking Score Performance Comparison', fontweight='bold', pad=15)
    
    # 实际TOP-1 docking scores数据
    methods = ['RGA', 'AutoGrow4.0', 'MARS', 'GEGL', 'REINVENT', 'FragGPT-GA']
    scores = [12.869, 12.474, 9.257, 10.450, 12.010, 13.458]  # 绝对值
    errors = [0.473, 0.839, 0.791, 1.040, 0.833, 0.442]
    
    # 颜色设置，使用统一的蓝绿色系，FragGPT-GA突出显示
    colors = ['#9DC3E6', '#9DC3E6', '#9DC3E6', '#9DC3E6', '#9DC3E6', '#2E8B57']
    
    bars = ax1.bar(range(len(methods)), scores, yerr=errors, capsize=4, 
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # 突出最佳结果
    bars[-1].set_linewidth(2.5)
    bars[-1].set_edgecolor('darkgreen')
    
    # 移除坐标轴
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # 添加方法名称作为标签
    for i, (bar, method) in enumerate(zip(bars, methods)):
        ax1.text(bar.get_x() + bar.get_width()/2., -0.5,
                 method, ha='center', va='top', rotation=45, fontsize=9)
    
    # 添加数值标签
    for i, (bar, score, err) in enumerate(zip(bars, scores, errors)):
        height = bar.get_height()
        fontweight = 'bold' if i == len(bars)-1 else 'normal'
        color = 'darkgreen' if i == len(bars)-1 else 'black'
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.1,
                 f'{score:.2f}', ha='center', va='bottom', 
                 fontweight=fontweight, color=color, fontsize=9)
    
    # === 图2: 多目标性能雷达图 ===
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.set_title('(B) Multi-objective Performance Profile', fontweight='bold', pad=20)
    
    # 选择代表性方法
    selected_methods = ['RGA', 'AutoGrow4.0', 'MARS', 'FragGPT-GA']
    
    # 标准化数据（0-1范围）
    docking_raw = [12.869, 12.474, 9.257, 13.458]
    qed_raw = [0.742, 0.748, 0.709, 0.764]
    sa_raw = [2.473, 2.497, 2.450, 2.014]  # 越小越好
    novelty_raw = [100, 100, 100, 100]  # 新颖性
    
    # 标准化
    docking_norm = [(x - min(docking_raw)) / (max(docking_raw) - min(docking_raw)) for x in docking_raw]
    qed_norm = [(x - min(qed_raw)) / (max(qed_raw) - min(qed_raw)) for x in qed_raw]
    sa_norm = [(max(sa_raw) - x) / (max(sa_raw) - min(sa_raw)) for x in sa_raw]  # 反转
    novelty_norm = [1.0] * 4  # 都是100%
    
    # 指标和角度
    metrics = ['Binding\nAffinity', 'Drug-likeness\n(QED)', 'Synthetic\nAccessibility', 'Novelty']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # 颜色 - 统一使用蓝绿色系
    colors_radar = ['#B7D7E8', '#9DC3E6', '#7BAFD4', '#2E8B57']
    
    for i, (method, color) in enumerate(zip(selected_methods, colors_radar)):
        values = [docking_norm[i], qed_norm[i], sa_norm[i], novelty_norm[i]]
        values += values[:1]  # 闭合
        
        linewidth = 3 if method == 'FragGPT-GA' else 2
        alpha_fill = 0.3 if method == 'FragGPT-GA' else 0.1
        
        ax2.plot(angles, values, 'o-', linewidth=linewidth, label=method, color=color)
        if method == 'FragGPT-GA':
            ax2.fill(angles, values, alpha=alpha_fill, color=color)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.grid(True)
    
    # === 图3: 动态掩码策略 ===
    ax3.set_title('(C) Dynamic Masking Strategy', fontweight='bold', pad=15)
    
    generations = np.arange(1, 26)
    n_initial = 3
    n_final = 1
    G_max = 25
    
    n_mask = n_initial + (generations - 1) / (G_max - 1) * (n_final - n_initial)
    
    ax3.plot(generations, n_mask, 'b-', linewidth=3, marker='o', markersize=5, 
             markerfacecolor='orange', markeredgecolor='blue', markeredgewidth=1.5)
    ax3.fill_between(generations, n_mask, alpha=0.2, color='lightblue')
    
    ax3.set_xlabel('Generation', fontweight='bold')
    ax3.set_ylabel('Number of Masked Fragments', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加阶段标注
    ax3.annotate('Exploration Phase', xy=(5, 2.7), xytext=(8, 2.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')
    ax3.annotate('Refinement Phase', xy=(20, 1.2), xytext=(17, 1.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, color='green', fontweight='bold')
    
    # 添加公式
    ax3.text(0.02, 0.98, r'$n_{mask}(g) = n_{initial} + \frac{g-1}{G_{max}-1}(n_{final} - n_{initial})$', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    # === 图4: 相交圆图 - FROGENT样式 ===
    ax4.set_title('(D) FragGPT-GA Framework Ecosystem', fontweight='bold', pad=15)
    
    # 移除坐标轴
    ax4.set_xlim(-1.3, 1.3)
    ax4.set_ylim(-1.3, 1.3)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    # 三个相交的大圆，模仿FROGENT的DATABASE/TOOL/MODEL布局
    main_circles = [
        # (center_x, center_y, radius, color, main_label)
        (-0.4, 0.4, 0.7, '#C5E0B4', 'GENETIC\nALGORITHM'),      # 左上 - 绿色
        (0.4, 0.4, 0.7, '#9DC3E6', 'GPT\nGENERATION'),         # 右上 - 蓝色  
        (0, -0.5, 0.7, '#F8CBAD', 'FRAGMENT\nOPTIMIZATION')    # 下方 - 橙色
    ]
    
    # 绘制相交的圆圈（无边框）
    for cx, cy, radius, color, main_label in main_circles:
        circle = Circle((cx, cy), radius, color=color, alpha=0.5, ec='none')
        ax4.add_patch(circle)
        
        # 在圆圈顶部添加主标签
        label_y = cy + radius * 0.7
        ax4.text(cx, label_y, main_label, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='black')
    
    # 在各个圆圈内添加椭圆形组件标签（模仿FROGENT的椭圆标签样式）
    components = [
        # GA圆圈内的组件
        (-0.6, 0.2, 'Population\nEvolution'),
        (-0.2, 0.6, 'Selection\nPressure'),
        (-0.6, 0.6, 'Crossover\nMutation'),
        
        # GPT圆圈内的组件  
        (0.6, 0.2, 'Chemical\nKnowledge'),
        (0.2, 0.6, 'Molecular\nGeneration'),
        (0.6, 0.6, 'Distribution\nExpansion'),
        
        # Fragment圆圈内的组件
        (-0.2, -0.7, 'BRICS\nDecomposition'),
        (0.2, -0.7, 'Multi-objective\nOptimization'),
        (0, -0.9, 'Synthetic\nAccessibility'),
        
        # 交集区域的组件
        (0, 0.3, 'Learned\nDiversity'),    # GA + GPT 交集
        (-0.2, -0.1, 'Fragment\nEvolution'), # GA + Fragment 交集  
        (0.2, -0.1, 'Intelligent\nSampling'), # GPT + Fragment 交集
    ]
    
    for x, y, label in components:
        # 使用椭圆形标签，模仿FROGENT样式
        ax4.text(x, y, label, ha='center', va='center', fontsize=7, 
                fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.25", facecolor='white', 
                         edgecolor='gray', linewidth=1, alpha=0.9))
    
    # 在中心交集区域添加核心标签
    ax4.text(0, 0, 'FragGPT-GA\nCore', ha='center', va='center', 
             fontsize=11, fontweight='bold', color='#2E8B57',
             bbox=dict(boxstyle="round,pad=0.35", facecolor='white', 
                      edgecolor='#2E8B57', linewidth=2, alpha=0.95))
    
    # 添加外围功能标签
    outer_labels = [
        ('Dynamic Masking\nStrategy', 0, 1.1),
        ('NSGA-II Multi-objective\nSelection', 1.1, 0),
        ('Molecular Docking\nEvaluation', 0, -1.2),
        ('Scaffold Hopping\nCapability', -1.1, 0)
    ]
    
    for label, x, y in outer_labels:
        ax4.text(x, y, label, ha='center', va='center', fontsize=8, 
                fontweight='bold', color='#2E8B57',
                bbox=dict(boxstyle="round,pad=0.25", facecolor='#E8F4FD', 
                         edgecolor='#2E8B57', linewidth=1, alpha=0.8))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 生成专业图表
    fig = create_fraggpt_professional_figure()
    
    # 保存高质量图片
    output_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/fraggpt_ga_academic.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    print(f"学术级FragGPT-GA展示图已保存至: {output_path}")
    
    # 保存EPS格式
    eps_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/fraggpt_ga_academic.eps"
    fig.savefig(eps_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='eps')
    print(f"EPS格式已保存至: {eps_path}")
    
    plt.show()
