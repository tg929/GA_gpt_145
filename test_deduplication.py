#!/usr/bin/env python3
"""
测试去重功能的脚本
验证修改后的多目标选择是否能正确去重
"""

import sys
import os
import subprocess
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def count_duplicates_in_file(file_path):
    """计算文件中的重复分子数量"""
    if not os.path.exists(file_path):
        return 0, 0, 0
    
    with open(file_path, 'r') as f:
        lines = [line.strip().split('\t') for line in f if line.strip()]
    
    smiles_list = [line[0] for line in lines]
    total_count = len(smiles_list)
    unique_count = len(set(smiles_list))
    duplicate_count = total_count - unique_count
    
    return total_count, unique_count, duplicate_count

def test_generation_deduplication(output_dir):
    """测试特定输出目录中的去重情况"""
    print(f"\n=== 测试目录: {output_dir} ===")
    
    # 查找最新的generation目录
    gen_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith('generation_') and item != 'generation_0':
            try:
                gen_num = int(item.split('_')[1])
                gen_dirs.append((gen_num, item))
            except ValueError:
                continue
    
    if not gen_dirs:
        print("未找到generation目录")
        return
    
    # 选择最大的generation数
    latest_gen_num, latest_gen_dir = max(gen_dirs)
    gen_path = os.path.join(output_dir, latest_gen_dir)
    
    print(f"检查最新generation: {latest_gen_dir}")
    
    # 检查种群文件
    population_file = os.path.join(gen_path, "initial_population_docked.smi")
    if os.path.exists(population_file):
        total, unique, duplicates = count_duplicates_in_file(population_file)
        print(f"当前种群文件: {population_file}")
        print(f"  总分子数: {total}")
        print(f"  唯一分子数: {unique}")
        print(f"  重复分子数: {duplicates}")
        if total > 0:
            print(f"  重复率: {duplicates/total*100:.1f}%")
    else:
        print(f"未找到种群文件: {population_file}")
    
    # 检查其他相关文件
    other_files = [
        "offspring_docked.smi",
        "current_parent_smiles.smi",
        "ga_input_pool.smi"
    ]
    
    for filename in other_files:
        filepath = os.path.join(gen_path, filename)
        if os.path.exists(filepath):
            total, unique, duplicates = count_duplicates_in_file(filepath)
            print(f"{filename}: 总数={total}, 唯一={unique}, 重复={duplicates}")

def main():
    print("=== 去重功能测试脚本 ===")
    
    # 测试样例目录
    test_dirs = [
        "output_gpt_multi_initial100/4r6e",
        "output_gpt_multi_nap/4r6e"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            test_generation_deduplication(test_dir)
        else:
            print(f"目录不存在: {test_dir}")
    
    print("\n=== 测试完成 ===")
    print("如果想要重新运行一代进化测试去重效果，可以使用以下命令:")
    print("python GA_main.py --config GA_gpt/config_GA_gpt.json --initial_population_file your_initial_file.smi --receptor 4r6e --max_generations 1")

if __name__ == "__main__":
    main()
