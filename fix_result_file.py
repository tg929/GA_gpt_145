#!/usr/bin/env python3
"""
脚本用于修改result_1iep.smi文件
- 删除ID列（第二列）
- 只保留SMILES和三个分数列
- 输出格式：SMILES\t分数1\t分数2\t分数3
"""

def fix_smi_file(input_file, output_file=None):
    """
    修复SMI文件格式
    - 删除ID列（第二列）
    - 只保留SMILES和三个分数
    """
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # 按制表符分割
        parts = line.strip().split('\t')
        
        if len(parts) >= 5:
            # 保留字段：SMILES(0), 分数1(2), 分数2(3), 分数3(4)，跳过ID(1)
            selected_parts = [parts[0], parts[2], parts[3], parts[4]]
            fixed_line = '\t'.join(selected_parts)
            fixed_lines.append(fixed_line + '\n')
        elif len(parts) >= 4:
            # 如果只有4个字段（可能已经没有ID了），直接保留
            fixed_line = '\t'.join(parts[:4])
            fixed_lines.append(fixed_line + '\n')
        else:
            # 如果字段数不够，保持原样
            fixed_lines.append(line)
    
    # 写入修复后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"处理完成！修复了 {len(lines)} 行")
    print(f"删除了ID列，只保留SMILES和三个分数")
    print(f"输出格式：SMILES\\t分数1\\t分数2\\t分数3")

if __name__ == "__main__":
    input_file = "/data1/ytg/medium_models/GA_gpt/result_1iep.smi"
    fix_smi_file(input_file)
    print("文件修复完成！")
