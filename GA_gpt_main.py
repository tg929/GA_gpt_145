#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragGPT-GA: 混合分子生成项目主入口
"""
import os
import sys
import argparse
import logging
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# --- 项目根目录设置 ---
# 此脚本位于项目根目录下，直接获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

# --- 模块导入 ---
from operations.operations_execute_GAgpt_finetune import GAGPTWorkflowExecutor
from utils.cpu_utils import get_available_cpu_cores, calculate_optimal_workers

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GA_GPT_MAIN")

def run_workflow_for_receptor(config_path: str, receptor_name: str, output_dir: str, num_processors: int) -> Tuple[str, bool]:
    """
    为单个受体运行完整工作流的包装函数，用于并行处理。    
    Args:
        config_path: 配置文件路径
        receptor_name: 受体名称(可以为None表示默认受体)
        output_dir: 输出目录
        num_processors: 分配给该进程的CPU核心数    
    Returns:
        Tuple[str, bool]: (受体显示名称, 是否成功)
    """
    # 使用显示名称，方便日志中识别默认受体
    receptor_display_name = receptor_name if receptor_name else "default"    
    logger.info("=" * 80)
    logger.info(f"启动子进程，为受体 '{receptor_display_name}' 运行GA-GPT混合工作流")
    logger.info(f"分配的CPU核心数: {num_processors}")
    logger.info(f"进程ID: {os.getpid()}")
    logger.info("=" * 80)
    try:
        # 初始化工作流执行器，并传入为该进程分配的处理器数量
        executor = GAGPTWorkflowExecutor(
            config_path=config_path, 
            receptor_name=receptor_name,
            output_dir_override=output_dir,
            num_processors_override=num_processors
        )        
        # 运行完整的工作流
        success = executor.run_complete_workflow()        
        if success:
            logger.info(f"子进程成功完成: 受体 '{receptor_display_name}' (PID: {os.getpid()})")
        else:
            logger.error(f"子进程失败: 受体 '{receptor_display_name}' (PID: {os.getpid()})")            
        return receptor_display_name, success
            
    except Exception as e:
        logger.critical(f"为受体 '{receptor_display_name}' 运行子流程时发生未捕获的严重异常: {e}", exc_info=True)
        return receptor_display_name, False

def main():
    """
    主函数:解析参数,启动GA-GPT工作流。
    所有并行控制完全由配置文件决定，无需命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="GA-GPT 混合分子生成项目主入口",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='GA_gpt/config_GA_gpt.json', help='主配置文件的路径')
    parser.add_argument('--receptor', type=str, default=None, help='(可选) 指定要运行的目标受体名称')
    parser.add_argument('--all_receptors', action='store_true', help='(可选) 运行配置文件中target_list的所有受体')
    parser.add_argument('--output_dir', type=str, default=None, help='(可选) 指定输出总目录')

    args = parser.parse_args()

    # --- 1. 加载配置并确定要运行的受体列表 ---
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        logger.critical(f"配置文件解析失败: {args.config}")
        sys.exit(1)        
    receptors_to_run = []
    if args.all_receptors:
        logger.info("检测到 --all_receptors 标志，将为配置文件中的所有受体运行工作流")
        receptors_to_run = list(config.get('receptors', {}).get('target_list', {}).keys())
    else:#自定义的受体配置
        receptors_to_run.append(args.receptor)
    logger.info(f"计划运行的受体列表: {receptors_to_run}")

    # --- 2. 从配置文件读取并行设置 ---
    performance_config = config.get('performance', {})
    parallel_enabled = performance_config.get('parallel_processing')
    max_workers_config = performance_config.get('max_workers')
    inner_processors_config = performance_config.get('number_of_processors')
    
    num_receptors = len(receptors_to_run)
    
    logger.info(f"配置文件并行设置:")
    logger.info(f"  - 并行处理: {'启用' if parallel_enabled else '禁用'}")
    logger.info(f"  - max_workers: {max_workers_config}")
    logger.info(f"  - number_of_processors: {inner_processors_config}")

    if not parallel_enabled or num_receptors <= 1:
        # 串行执行模式
        logger.info("=" * 60)
        logger.info("使用串行执行模式")
        logger.info("=" * 60)
        
        # 即使是串行，也要检测可用核心用于受体内部并行
        if inner_processors_config == -1:
            available_cores, cpu_usage = get_available_cpu_cores()
            cores_per_receptor = available_cores
        else:
            cores_per_receptor = inner_processors_config
            
        logger.info(f"单受体使用CPU核心数: {cores_per_receptor}")
        
        successful_runs = []
        failed_runs = []
        
        for receptor_name in receptors_to_run:
            receptor_display_name, success = run_workflow_for_receptor(
                args.config, receptor_name, args.output_dir, cores_per_receptor
            )
            if success:
                successful_runs.append(receptor_display_name)
            else:
                failed_runs.append(receptor_display_name)
    else:
        # 并行执行模式
        logger.info("=" * 60)
        logger.info("使用并行执行模式")
        logger.info("正在检测系统可用CPU资源...")
        logger.info("=" * 60)
        
        # 动态检测可用CPU资源
        available_cores, cpu_usage = get_available_cpu_cores()
        
        # 计算最优并行配置
        if max_workers_config == -1 and inner_processors_config == -1:
            # 全自动模式
            max_workers, cores_per_worker = calculate_optimal_workers(
                target_count=num_receptors,
                available_cores=available_cores,
                cores_per_worker=-1
            )
        elif max_workers_config == -1:
            # 受体间自动，受体内固定
            max_possible_workers = available_cores // inner_processors_config
            max_workers = min(num_receptors, max_possible_workers)
            cores_per_worker = inner_processors_config
        elif inner_processors_config == -1:
            # 受体间固定，受体内自动
            max_workers = min(max_workers_config, num_receptors)
            cores_per_worker = max(1, available_cores // max_workers)
        else:
            # 全手动模式
            max_workers = min(max_workers_config, num_receptors)
            cores_per_worker = inner_processors_config
        
        logger.info(f"并行执行配置:")
        logger.info(f"  - 同时运行的受体数: {max_workers}")
        logger.info(f"  - 每个受体CPU核心数: {cores_per_worker}")
        logger.info(f"  - 预估总使用核心数: {max_workers * cores_per_worker}")
        logger.info(f"  - 当前系统CPU使用率: {cpu_usage:.1f}%")
        
        successful_runs = []
        failed_runs = []
        
        # 启动并行执行,增加超时和死锁保护
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_workflow_for_receptor,
                    args.config,
                    receptor_name,
                    args.output_dir,
                    cores_per_worker
                ): receptor_name for receptor_name in receptors_to_run
            }
            
            # 添加超时机制，防止进程无限等待
            import concurrent.futures
            try:
                # 使用 as_completed 来获取已完成的结果
                for future in as_completed(futures, timeout=7200):  # 2小时总超时
                    receptor_name = futures[future]
                    try:
                        # 获取结果，这里可以设置一个短超时，但通常不是必须的
                        receptor_display_name, success = future.result()
                        if success:
                            successful_runs.append(receptor_display_name)
                        else:
                            failed_runs.append(receptor_display_name)
                    except Exception as exc:
                        logger.error(f"受体 '{receptor_name}' 在执行期间生成了异常: {exc}", exc_info=True)
                        failed_runs.append(receptor_name)

            except concurrent.futures.TimeoutError:
                logger.error("并行执行总超时(2小时),可能存在死锁。正在强制终止所有剩余的子进程...")
                # 强制关闭进程池并取消所有正在运行和等待的任务
                # cancel_futures=True 是Python 3.9+的新特性，可以终止正在运行的任务
                executor.shutdown(wait=False, cancel_futures=True)
                
                # 记录所有尚未完成的任务为失败
                for future, receptor_name in futures.items():
                    if not future.done():
                        # 不需要调用 future.cancel()，因为 shutdown 已经处理了
                        failed_runs.append(receptor_name if receptor_name else "default")
            
            # 再次检查并确保所有未记录的结果都被标记
            completed_receptors = set(successful_runs + failed_runs)
            all_receptors = set(receptors_to_run)
            missing_receptors = all_receptors - completed_receptors
            if missing_receptors:
                logger.warning(f"以下受体未返回明确结果，标记为失败: {list(missing_receptors)}")
                failed_runs.extend(list(missing_receptors))


    # --- 3. 最终总结报告 ---
    logger.info("=" * 80)
    logger.info("所有GA-GPT工作流执行完毕")
    logger.info(f"成功运行的受体 ({len(successful_runs)}): {successful_runs}")
    logger.info(f"失败的受体 ({len(failed_runs)}): {failed_runs}")
    logger.info("=" * 80)

    if failed_runs:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    # 在Windows或macOS上，有必要将multiprocessing的启动方法设置为'spawn'或'forkserver'
    # 对于Linux, 'fork'通常是默认且可以工作的，但'spawn'更安全。
    # 为防止在多线程+多进程混合编程中出现死锁（如此次遇到的情况），
    # 我们统一将启动方法强制设置为'spawn'，以保证在所有平台上的稳定运行。
    multiprocessing.set_start_method('spawn', force=True)
    
    main()
