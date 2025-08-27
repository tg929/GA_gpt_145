#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文专用雷达图生成器
===================
生成适合论文首页使用的简洁雷达图
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# 设置专业的学术风格
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 9

def create_paper_radar_chart():
    """创建适合论文的简洁雷达图"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(3.2, 2.8), subplot_kw=dict(projection='polar'))
    
    # 包含论文表格中的所有方法
    methods_data = {
        'screening': {
            'docking': 11.400,
            'qed': 0.678,
            'sa': 2.689,
            'novelty': 0.0,
            'color': '#FFB6C1',
            'linestyle': ':',
            'alpha': 0.6
        },
        'MARS': {
            'docking': 9.257,
            'qed': 0.709,
            'sa': 2.450,
            'novelty': 100.0,
            'color': '#87CEEB',
            'linestyle': '--',
            'alpha': 0.7
        },
        'MolDQN': {
            'docking': 7.501,
            'qed': 0.170,
            'sa': 5.833,
            'novelty': 100.0,
            'color': '#DDA0DD',
            'linestyle': ':',
            'alpha': 0.6
        },
        'GEGL': {
            'docking': 10.450,
            'qed': 0.643,
            'sa': 2.990,
            'novelty': 100.0,
            'color': '#F0E68C',
            'linestyle': '-.',
            'alpha': 0.7
        },
        'REINVENT': {
            'docking': 12.010,
            'qed': 0.445,
            'sa': 2.596,
            'novelty': 100.0,
            'color': '#FFA07A',
            'linestyle': '--',
            'alpha': 0.7
        },
        'RationaleRL': {
            'docking': 11.642,
            'qed': 0.315,
            'sa': 2.919,
            'novelty': 100.0,
            'color': '#98FB98',
            'linestyle': ':',
            'alpha': 0.6
        },
        'JTVAE': {
            'docking': 10.963,
            'qed': 0.593,
            'sa': 3.222,
            'novelty': 98.0,
            'color': '#D3D3D3',
            'linestyle': '-.',
            'alpha': 0.6
        },
        'Gen3D': {
            'docking': 9.832,
            'qed': 0.701,
            'sa': 3.450,
            'novelty': 100.0,
            'color': '#B0C4DE',
            'linestyle': ':',
            'alpha': 0.6
        },
        'GA+D': {
            'docking': 8.760,
            'qed': 0.405,
            'sa': 5.024,
            'novelty': 99.0,
            'color': '#F5DEB3',
            'linestyle': '--',
            'alpha': 0.6
        },
        'Graph-GA': {
            'docking': 12.302,
            'qed': 0.456,
            'sa': 3.503,
            'novelty': 100.0,
            'color': '#20B2AA',
            'linestyle': '-.',
            'alpha': 0.7
        },
        'AutoGrow4.0': {
            'docking': 12.474,
            'qed': 0.748,
            'sa': 2.497,
            'novelty': 100.0,
            'color': '#9DC3E6',
            'linestyle': '--',
            'alpha': 0.8
        },
        'RGA': {
            'docking': 12.869,
            'qed': 0.742,
            'sa': 2.473,
            'novelty': 100.0,
            'color': '#C5E0B4',
            'linestyle': '-',
            'alpha': 0.8
        },
        'FragGPT-GA': {
            'docking': 13.458,
            'qed': 0.764,
            'sa': 2.014,
            'novelty': 100.0,
            'color': '#2E8B57',
            'linestyle': '-',
            'alpha': 1.0
        }
    }
    
    # 指标名称
    metrics = ['Binding\nAffinity', 'Drug-likeness\n(QED)', 
               'Synthetic   \nAccessibility       ', 'Novelty']
    
    # 计算角度
    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 数据标准化
    all_docking = [data['docking'] for data in methods_data.values()]
    all_qed = [data['qed'] for data in methods_data.values()]
    all_sa = [data['sa'] for data in methods_data.values()]
    all_novelty = [data['novelty'] for data in methods_data.values()]
    
    min_docking, max_docking = min(all_docking), max(all_docking)
    min_qed, max_qed = min(all_qed), max(all_qed)
    min_sa, max_sa = min(all_sa), max(all_sa)
    min_novelty, max_novelty = min(all_novelty), max(all_novelty)
    
    # 绘制每个方法
    for method_name, data in methods_data.items():
        # 标准化数据
        norm_docking = (data['docking'] - min_docking) / (max_docking - min_docking) if max_docking != min_docking else 0.5
        norm_qed = (data['qed'] - min_qed) / (max_qed - min_qed) if max_qed != min_qed else 0.5
        norm_sa = (max_sa - data['sa']) / (max_sa - min_sa) if max_sa != min_sa else 0.5  # 反转SA（越小越好）
        norm_novelty = (data['novelty'] - min_novelty) / (max_novelty - min_novelty) if max_novelty != min_novelty else 0.5
        
        values = [norm_docking, norm_qed, norm_sa, norm_novelty]
        values += values[:1]
        
        # 设置线条样式
        linewidth = 2.5 if method_name == 'FragGPT-GA' else 1.0
        alpha_line = data.get('alpha', 0.7)
        alpha_fill = 0.2 if method_name == 'FragGPT-GA' else 0
        markersize = 4 if method_name == 'FragGPT-GA' else 2
        
        # 绘制线条
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
                label=method_name, color=data['color'],
                linestyle=data['linestyle'], alpha=alpha_line, 
                markersize=markersize)
        
        # 为FragGPT-GA添加填充，使用指定的RGB颜色
        if method_name == 'FragGPT-GA':
            fill_color = (197/255, 224/255, 180/255)  # RGB(197, 224, 180)
            ax.fill(angles, values, alpha=alpha_fill, color=fill_color)
    
    # 设置角度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=7, fontweight='normal')
    
    # 设置径向轴
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # 添加图例（只显示关键方法，避免过于拥挤）
    # 筛选出关键方法用于图例显示
    key_methods = ['RGA', 'AutoGrow4.0', 'MARS', 'GEGL', 'REINVENT', 'FragGPT-GA']
    handles, labels = ax.get_legend_handles_labels()
    
    # 只保留关键方法的图例
    key_handles = []
    key_labels = []
    for handle, label in zip(handles, labels):
        if label in key_methods:
            key_handles.append(handle)
            key_labels.append(label)
    
    ax.legend(key_handles, key_labels, loc='upper right', bbox_to_anchor=(1.3, 1.15), 
              fontsize=5, frameon=True, fancybox=True, shadow=False, ncol=1)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 生成论文雷达图
    fig = create_paper_radar_chart()
    
    # 保存为论文格式
    output_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/paper_radar_chart.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    print(f"论文雷达图已保存至: {output_path}")
    
    # 保存EPS格式用于LaTeX
    eps_path = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/paper_radar_chart.eps"
    fig.savefig(eps_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='eps')
    print(f"论文雷达图EPS格式已保存至: {eps_path}")
    
    plt.show()
