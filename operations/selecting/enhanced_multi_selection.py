#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的多目标选择策略
解决帕累托前沿"凹陷"问题的高级选择算法
"""
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def identify_sparse_regions(objectives: np.ndarray, fronts: List[List[int]]) -> List[int]:
    """
    识别帕累托前沿中的稀疏区域
    
    Args:
        objectives: 目标函数矩阵
        fronts: 帕累托前沿列表
        
    Returns:
        稀疏区域的分子索引列表
    """
    if not fronts or not fronts[0]:
        return []
    
    first_front = fronts[0]
    if len(first_front) < 3:
        return first_front
    
    # 计算第一前沿中每个点到其邻居的平均距离
    front_objectives = objectives[first_front]
    distances = []
    
    for i, point in enumerate(front_objectives):
        # 计算到其他所有点的欧几里得距离
        other_points = np.delete(front_objectives, i, axis=0)
        point_distances = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
        # 使用最近邻距离作为稀疏度指标
        min_distance = np.min(point_distances) if len(point_distances) > 0 else 0
        distances.append(min_distance)
    
    # 识别距离大于平均值的点作为稀疏区域
    mean_distance = np.mean(distances)
    sparse_indices = [first_front[i] for i, d in enumerate(distances) if d > mean_distance]
    
    logger.info(f"识别到 {len(sparse_indices)} 个稀疏区域的分子")
    return sparse_indices

def enhanced_nsga2_selection(molecules_data: List[Dict], n_select: int, 
                           sparse_region_bonus: float = 0.2) -> List[Dict]:
    """
    增强的NSGA-II选择，优先选择稀疏区域的分子
    
    Args:
        molecules_data: 分子数据列表
        n_select: 要选择的分子数量
        sparse_region_bonus: 稀疏区域分子的选择奖励
        
    Returns:
        选中的分子列表
    """
    if not molecules_data or n_select <= 0:
        return []
    
    # 构建目标矩阵
    objectives = np.array([
        [m['docking_score'], -m.get('qed_score', 0), m.get('sa_score', 10)] 
        for m in molecules_data
    ])
    
    # 执行标准NSGA-II
    from operations.selecting.selecting_multi_demo import non_dominated_sort, crowding_distance
    fronts = non_dominated_sort(objectives)
    
    # 识别稀疏区域
    sparse_indices = identify_sparse_regions(objectives, fronts)
    
    selected_molecules = []
    selected_indices = set()
    
    # 优先选择稀疏区域的分子
    sparse_selected = min(len(sparse_indices), max(1, int(n_select * sparse_region_bonus)))
    for i in range(sparse_selected):
        if i < len(sparse_indices):
            idx = sparse_indices[i]
            selected_molecules.append(molecules_data[idx])
            selected_indices.add(idx)
    
    logger.info(f"优先选择了 {len(selected_molecules)} 个稀疏区域分子")
    
    # 用标准NSGA-II选择剩余分子
    remaining_needed = n_select - len(selected_molecules)
    if remaining_needed > 0:
        # 从未选择的分子中继续选择
        remaining_molecules = [mol for i, mol in enumerate(molecules_data) 
                             if i not in selected_indices]
        
        if remaining_molecules:
            # 重新计算目标矩阵
            remaining_objectives = np.array([
                [m['docking_score'], -m.get('qed_score', 0), m.get('sa_score', 10)] 
                for m in remaining_molecules
            ])
            
            remaining_fronts = non_dominated_sort(remaining_objectives)
            
            for front in remaining_fronts:
                if len(selected_molecules) + len(front) <= n_select:
                    selected_molecules.extend([remaining_molecules[i] for i in front])
                else:
                    # 使用拥挤度距离选择
                    distances = crowding_distance(remaining_objectives, front)
                    sorted_by_crowding = sorted(zip(front, distances), 
                                              key=lambda x: x[1], reverse=True)
                    
                    remaining_needed = n_select - len(selected_molecules)
                    for i in range(remaining_needed):
                        if i < len(sorted_by_crowding):
                            idx = sorted_by_crowding[i][0]
                            selected_molecules.append(remaining_molecules[idx])
                    break
    
    logger.info(f"增强选择完成: 共选择 {len(selected_molecules)} 个分子")
    return selected_molecules

def objective_guided_selection(molecules_data: List[Dict], n_select: int) -> List[Dict]:
    """
    目标引导选择：确保在目标空间的关键区域都有代表
    
    Args:
        molecules_data: 分子数据列表  
        n_select: 要选择的分子数量
        
    Returns:
        选中的分子列表
    """
    if not molecules_data or n_select <= 0:
        return []
    
    # 定义目标空间的关键区域
    regions = {
        'high_docking': {'weight': 0.3, 'criteria': lambda m: m['docking_score'] < -9},
        'high_qed': {'weight': 0.25, 'criteria': lambda m: m.get('qed_score', 0) > 0.7},
        'low_sa': {'weight': 0.25, 'criteria': lambda m: m.get('sa_score', 10) < 3},
        'balanced': {'weight': 0.2, 'criteria': lambda m: (
            m['docking_score'] < -7 and 
            m.get('qed_score', 0) > 0.5 and 
            m.get('sa_score', 10) < 5
        )}
    }
    
    selected_molecules = []
    used_indices = set()
    
    # 为每个区域分配分子
    for region_name, region_config in regions.items():
        target_count = int(n_select * region_config['weight'])
        region_molecules = []
        
        for i, mol in enumerate(molecules_data):
            if i not in used_indices and region_config['criteria'](mol):
                region_molecules.append((i, mol))
        
        # 从该区域选择最好的分子
        if region_name == 'high_docking':
            region_molecules.sort(key=lambda x: x[1]['docking_score'])
        elif region_name == 'high_qed':
            region_molecules.sort(key=lambda x: x[1].get('qed_score', 0), reverse=True)
        elif region_name == 'low_sa':
            region_molecules.sort(key=lambda x: x[1].get('sa_score', 10))
        else:  # balanced
            region_molecules.sort(key=lambda x: (
                x[1]['docking_score'] + 
                (-x[1].get('qed_score', 0)) + 
                x[1].get('sa_score', 10)
            ))
        
        # 选择该区域的分子
        selected_count = 0
        for idx, mol in region_molecules:
            if selected_count >= target_count:
                break
            selected_molecules.append(mol)
            used_indices.add(idx)
            selected_count += 1
        
        logger.info(f"区域 {region_name}: 选择了 {selected_count} 个分子")
    
    # 如果还没选够，用NSGA-II补充
    if len(selected_molecules) < n_select:
        remaining_molecules = [mol for i, mol in enumerate(molecules_data) 
                             if i not in used_indices]
        remaining_needed = n_select - len(selected_molecules)
        
        if remaining_molecules:
            additional = enhanced_nsga2_selection(remaining_molecules, remaining_needed)
            selected_molecules.extend(additional)
    
    logger.info(f"目标引导选择完成: 共选择 {len(selected_molecules)} 个分子")
    return selected_molecules