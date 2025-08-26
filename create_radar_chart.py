#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragGPT-GA多方法对比雷达图
===========================
基于论文表格数据的完整方法对比雷达图
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# 设置专业的学术风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18

def create_comprehensive_radar_chart():
    """创建包含所有方法的多目标性能雷达图"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Multi-objective Performance Comparison: FragGPT-GA vs Baselines', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 所有方法及其数据（基于论文表格）
    methods_data = {
        'RGA': {
            'docking': 12.869,
            'qed': 0.742,
            'sa': 2.473,
            'novelty': 100.0,
            'color': '#FF6B6B',
            'linestyle': '-'
        },
        'AutoGrow4.0': {
            'docking': 12.474,
            'qed': 0.748,
            'sa': 2.497,
            'novelty': 100.0,
            'color': '#4ECDC4',
            'linestyle': '--'
        },
        'MARS': {
            'docking': 9.257,
            'qed': 0.709,
            'sa': 2.450,
            'novelty': 100.0,
            'color': '#45B7D1',
            'linestyle': '-.'
        },
        'GEGL': {
            'docking': 10.450,
            'qed': 0.643,
            'sa': 2.990,
            'novelty': 100.0,
            'color': '#96CEB4',
            'linestyle': ':'
        },
        'REINVENT': {
            'docking': 12.010,
            'qed': 0.445,
            'sa': 2.596,
            'novelty': 100.0,
            'color': '#FFEAA7',
            'linestyle': '-'
        },
        'MolDQN': {
            'docking': 11.215,
            'qed': 0.735,
            'sa': 2.380,
            'novelty': 100.0,
            'color': '#DDA0DD',
            'linestyle': '--'
        },
        'RationaleRL': {
            'docking': 10.840,
            'qed': 0.720,
            'sa': 2.420,
            'novelty': 100.0,
            'color': '#F4A460',
            'linestyle': '-.'
        },
        'JTVAE': {
            'docking': 9.680,
            'qed': 0.690,
            'sa': 2.510,
            'novelty': 100.0,
            'color': '#87CEEB',
            'linestyle': ':'
        },
        'Gen3D': {
            'docking': 8.950,
            'qed': 0.665,
            'sa': 2.650,
            'novelty': 100.0,
            'color': '#DAA520',
            'linestyle': '-'
        },
        'GA+D': {
            'docking': 11.580,
            'qed': 0.710,
            'sa': 2.390,
            'novelty': 100.0,
            'color': '#CD853F',
            'linestyle': '--'
        },
        'Graph-GA': {
            'docking': 10.920,
            'qed': 0.695,
            'sa': 2.445,
            'novelty': 100.0,
            'color': '#20B2AA',
            'linestyle': '-.'
        },
        'FragGPT-GA': {
            'docking': 13.458,
            'qed': 0.764,
            'sa': 2.014,
            'novelty': 100.0,
            'color': '#2E8B57',
            'linestyle': '-'
        }
    }
    
    # 指标名称
    metrics = ['Binding Affinity\n(|kcal/mol|)', 'Drug-likeness\n(QED)', 
               'Synthetic Accessibility\n(Reversed)', 'Novelty\n(%)']
    
    # 计算角度
    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 数据标准化
    # 获取所有数值用于标准化
    all_docking = [data['docking'] for data in methods_data.values()]
    all_qed = [data['qed'] for data in methods_data.values()]
    all_sa = [data['sa'] for data in methods_data.values()]
    
    min_docking, max_docking = min(all_docking), max(all_docking)
    min_qed, max_qed = min(all_qed), max(all_qed)
    min_sa, max_sa = min(all_sa), max(all_sa)
    
    # 绘制每个方法
    for method_name, data in methods_data.items():
        # 标准化数据 (0-1 范围)
        norm_docking = (data['docking'] - min_docking) / (max_docking - min_docking)
        norm_qed = (data['qed'] - min_qed) / (max_qed - min_qed)
        norm_sa = (max_sa - data['sa']) / (max_sa - min_sa)  # 反转SA（越小越好）
        norm_novelty = 1.0  # 所有方法都是100%
        
        values = [norm_docking, norm_qed, norm_sa, norm_novelty]
        values += values[:1]  # 闭合
        
        # 设置线条样式
        linewidth = 4 if method_name == 'FragGPT-GA' else 2
        alpha_line = 0.9 if method_name == 'FragGPT-GA' else 0.7
        alpha_fill = 0.3 if method_name == 'FragGPT-GA' else 0.1
        
        # 绘制线条和填充
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
                label=method_name, color=data['color'],
                linestyle=data['linestyle'], alpha=alpha_line, markersize=6)
        
        # 为FragGPT-GA添加填充
        if method_name == 'FragGPT-GA':
            ax.fill(angles, values, alpha=alpha_fill, color=data['color'])
    
    # 设置角度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    # 设置径向轴
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    # 添加数据说明
    data_info = """
    Data Range:
    • Binding Affinity: {:.1f} - {:.1f} |kcal/mol|
    • QED: {:.3f} - {:.3f}
    • SA: {:.3f} - {:.3f} (lower is better)
    • Novelty: 100% (all methods)
    """.format(min_docking, max_docking, min_qed, max_qed, min_sa, max_sa)
    
    plt.figtext(0.02, 0.02, data_info, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_focused_radar_chart():
    """创建重点方法对比雷达图（选择最具代表性的方法）"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    fig.suptitle('Key Methods Performance Comparison', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 选择代表性方法
    selected_methods = {
        'RGA': {
            'docking': 12.869,
            'qed': 0.742,
            'sa': 2.473,
            'novelty': 100.0,
            'color': '#FF6B6B',
            'linestyle': '-'
        },
        'AutoGrow4.0': {
            'docking': 12.474,
            'qed': 0.748,
            'sa': 2.497,
            'novelty': 100.0,
            'color': '#4ECDC4',
            'linestyle': '--'
        },
        'MARS': {
            'docking': 9.257,
            'qed': 0.709,
            'sa': 2.450,
            'novelty': 100.0,
            'color': '#45B7D1',
            'linestyle': '-.'
        },
        'GEGL': {
            'docking': 10.450,
            'qed': 0.643,
            'sa': 2.990,
            'novelty': 100.0,
            'color': '#96CEB4',
            'linestyle': ':'
        },
        'REINVENT': {
            'docking': 12.010,
            'qed': 0.445,
            'sa': 2.596,
            'novelty': 100.0,
            'color': '#FFEAA7',
            'linestyle': '-'
        },
        'FragGPT-GA': {
            'docking': 13.458,
            'qed': 0.764,
            'sa': 2.014,
            'novelty': 100.0,
            'color': '#2E8B57',
            'linestyle': '-'
        }
    }
    
    # 指标名称
    metrics = ['Binding Affinity\n(|kcal/mol|)', 'Drug-likeness\n(QED)', 
               'Synthetic Accessibility\n(Reversed)', 'Novelty\n(%)']
    
    # 计算角度
    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 数据标准化
    all_docking = [data['docking'] for data in selected_methods.values()]
    all_qed = [data['qed'] for data in selected_methods.values()]
    all_sa = [data['sa'] for data in selected_methods.values()]
    
    min_docking, max_docking = min(all_docking), max(all_docking)
    min_qed, max_qed = min(all_qed), max(all_qed)
    min_sa, max_sa = min(all_sa), max(all_sa)
    
    # 绘制每个方法
    for method_name, data in selected_methods.items():
        # 标准化数据
        norm_docking = (data['docking'] - min_docking) / (max_docking - min_docking)
        norm_qed = (data['qed'] - min_qed) / (max_qed - min_qed)
        norm_sa = (max_sa - data['sa']) / (max_sa - min_sa)
        norm_novelty = 1.0
        
        values = [norm_docking, norm_qed, norm_sa, norm_novelty]
        values += values[:1]
        
        # 设置线条样式
        linewidth = 4 if method_name == 'FragGPT-GA' else 2.5
        alpha_line = 0.9 if method_name == 'FragGPT-GA' else 0.8
        alpha_fill = 0.25 if method_name == 'FragGPT-GA' else 0.05
        markersize = 8 if method_name == 'FragGPT-GA' else 6
        
        # 绘制线条和填充
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
                label=method_name, color=data['color'],
                linestyle=data['linestyle'], alpha=alpha_line, markersize=markersize)
        
        if method_name == 'FragGPT-GA':
            ax.fill(angles, values, alpha=alpha_fill, color=data['color'])
    
    # 设置角度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    # 设置径向轴
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=11)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 生成完整对比雷达图
    print("生成完整方法对比雷达图...")
    fig1 = create_comprehensive_radar_chart()
    
    # 保存完整版
    output_path1 = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/comprehensive_radar_chart.png"
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    print(f"完整雷达图已保存至: {output_path1}")
    
    # 保存EPS格式
    eps_path1 = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/comprehensive_radar_chart.eps"
    fig1.savefig(eps_path1, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='eps')
    print(f"完整雷达图EPS格式已保存至: {eps_path1}")
    
    # 生成重点方法对比雷达图
    print("生成重点方法对比雷达图...")
    fig2 = create_focused_radar_chart()
    
    # 保存重点版
    output_path2 = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/focused_radar_chart.png"
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    print(f"重点雷达图已保存至: {output_path2}")
    
    # 保存EPS格式
    eps_path2 = "papers/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions/focused_radar_chart.eps"
    fig2.savefig(eps_path2, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='eps')
    print(f"重点雷达图EPS格式已保存至: {eps_path2}")
    
    plt.show()
