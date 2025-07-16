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
import signal
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

from operations.operations_execute_GAgpt_finetune import GAGPTWorkflowExecutor
from utils.cpu_utils import get_available_cpu_cores, calculate_optimal_workers

# 设置全局进程追踪字典
active_processes: Dict[int, Dict] = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GA_GPT_MAIN")

def monitor_process_memory(pid: int, process_name: str, interval: int = 60):
    """
    监控进程的内存使用情况
    
    Args:
        pid: 进程ID
        process_name: 进程名称
        interval: 监控间隔(秒)
    """
    try:
        process = psutil.Process(pid)
        while True:
            try:
                # 检查进程是否还存在
                if not psutil.pid_exists(pid) or process.status() == psutil.STATUS_ZOMBIE:
                    logger.info(f"进程 {process_name} (PID: {pid}) 已终止")
                    break
                
                # 获取内存使用情况
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)  # 转换为MB
                
                logger.debug(f"进程 {process_name} (PID: {pid}) 内存使用: {mem_mb:.2f} MB")
                
                # 如果内存使用过高，记录警告
                if mem_mb > 1000:  # 超过1GB
                    logger.warning(f"进程 {process_name} (PID: {pid}) 内存使用过高: {mem_mb:.2f} MB")
                
                time.sleep(interval)
            except psutil.NoSuchProcess:
                logger.info(f"进程 {process_name} (PID: {pid}) 已终止")
                break
            except Exception as e:
                logger.error(f"监控进程 {process_name} (PID: {pid}) 时出错: {e}")
                break
    except Exception as e:
        logger.error(f"启动监控进程 {process_name} (PID: {pid}) 时出错: {e}")

def cleanup_child_processes():
    """清理所有子进程"""
    current_pid = os.getpid()
    try:
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                logger.info(f"正在终止子进程 PID: {child.pid}")
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # 等待子进程终止
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        # 强制终止仍然存活的进程
        for p in alive:
            try:
                logger.warning(f"强制终止子进程 PID: {p.pid}")
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        logger.error(f"清理子进程时出错: {e}")

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
    
    # 注册进程信息
    process_info = {
        'receptor': receptor_display_name,
        'start_time': time.time(),
        'status': 'running'
    }
    active_processes[os.getpid()] = process_info
    
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
            process_info['status'] = 'completed'
        else:
            logger.error(f"子进程失败: 受体 '{receptor_display_name}' (PID: {os.getpid()})")
            process_info['status'] = 'failed'
            
        # 确保资源被释放
        executor.cleanup_resources()
        return receptor_display_name, success
            
    except Exception as e:
        logger.critical(f"为受体 '{receptor_display_name}' 运行子流程时发生未捕获的严重异常: {e}", exc_info=True)
        process_info['status'] = 'error'
        return receptor_display_name, False
    finally:
        # 记录结束时间
        process_info['end_time'] = time.time()
        duration = process_info['end_time'] - process_info['start_time']
        logger.info(f"受体 '{receptor_display_name}' 处理耗时: {duration:.2f} 秒")
        
        # 确保清理所有子进程
        try:
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    logger.info(f"终止子进程 PID: {child.pid}")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            logger.error(f"清理子进程时出错: {e}")

def calculate_optimal_workers_with_memory(target_count, available_cores, memory_per_worker=2.0):
    """
    计算最优的并行配置，考虑内存限制
    
    Args:
        target_count: 目标任务数量
        available_cores: 可用CPU核心数
        memory_per_worker: 每个工作进程估计需要的内存(GB)
        
    Returns:
        (max_workers, cores_per_worker): 最优工作进程数和每个进程的核心数
    """
    # 获取系统内存信息
    mem = psutil.virtual_memory()
    total_memory_gb = mem.total / (1024**3)  # 转换为GB
    available_memory_gb = mem.available / (1024**3)
    
    logger.info(f"系统总内存: {total_memory_gb:.1f}GB, 可用内存: {available_memory_gb:.1f}GB")
    
    # 基于内存限制计算最大工作进程数
    max_workers_by_memory = int(available_memory_gb / memory_per_worker)
    
    # 基于CPU核心的计算
    if target_count >= available_cores:
        # 任务数多于核心数，每个任务分配1个核心
        cores_per_worker = 1
        max_workers_by_cpu = min(target_count, available_cores)
    else:
        # 任务数少于核心数，平均分配
        cores_per_worker = available_cores // target_count
        max_workers_by_cpu = target_count
    
    # 综合CPU和内存限制
    max_workers = min(max_workers_by_cpu, max_workers_by_memory)
    max_workers = max(1, max_workers)  # 至少有1个工作进程
    
    logger.info(f"基于CPU限制的最大工作进程数: {max_workers_by_cpu}")
    logger.info(f"基于内存限制的最大工作进程数: {max_workers_by_memory}")
    logger.info(f"最终选择的工作进程数: {max_workers}, 每个进程核心数: {cores_per_worker}")
    
    return max_workers, cores_per_worker

def main():
    """
    主函数:解析参数,启动GA-GPT工作流。
    所有并行控制完全由配置文件决定，无需命令行参数。
    """
    # 设置信号处理，确保在程序被中断时清理资源
    def signal_handler(sig, frame):
        logger.info("接收到中断信号，正在清理资源...")
        cleanup_child_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="GA-GPT 混合分子生成项目主入口",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='GA_gpt/config_GA_gpt.json', help='主配置文件的路径')
    parser.add_argument('--receptor', type=str, default=None, help='(可选) 指定要运行的目标受体名称')
    parser.add_argument('--all_receptors', action='store_true', help='(可选) 运行配置文件中target_list的所有受体')
    parser.add_argument('--output_dir', type=str, default=None, help='(可选) 指定输出总目录')
    parser.add_argument('--memory_per_worker', type=float, default=2.0, help='(可选) 每个工作进程估计需要的内存(GB)')

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
    else:
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
        
        # 计算最优并行配置，考虑内存限制
        if max_workers_config == -1 and inner_processors_config == -1:
            # 全自动模式
            max_workers, cores_per_worker = calculate_optimal_workers_with_memory(
                target_count=num_receptors,
                available_cores=available_cores,
                memory_per_worker=args.memory_per_worker
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
        
        # 启动并行执行，增加超时和死锁防护
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for receptor_name in receptors_to_run:
                future = executor.submit(
                    run_workflow_for_receptor,
                    args.config,
                    receptor_name,
                    args.output_dir,
                    cores_per_worker
                )
                futures[future] = receptor_name
            
            # 添加超时机制，防止进程无限等待
            import concurrent.futures
            try:
                for future in as_completed(futures, timeout=7200):  # 2小时总超时
                    try:
                        receptor_display_name, success = future.result(timeout=60)  # 单个结果60秒超时
                        if success:
                            successful_runs.append(receptor_display_name)
                        else:
                            failed_runs.append(receptor_display_name)
                    except concurrent.futures.TimeoutError:
                        receptor_name = futures[future]
                        logger.error(f"获取受体 '{receptor_name}' 结果超时")
                        failed_runs.append(receptor_name)
                        # 尝试取消任务
                        future.cancel()
            except concurrent.futures.TimeoutError:
                logger.error("并行执行总超时，可能存在死锁。正在终止所有进程...")
                # 取消所有未完成的任务
                for future in futures:
                    if not future.done():
                        future.cancel()
                        receptor_name = futures[future]
                        failed_runs.append(receptor_name)
            finally:
                # 确保所有进程都被终止
                executor.shutdown(wait=False)
                # 额外清理可能残留的子进程
                cleanup_child_processes()

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
