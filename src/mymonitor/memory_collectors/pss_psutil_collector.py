import psutil
import time
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

from .base import AbstractMemoryCollector, ProcessMemorySample

logger = logging.getLogger(__name__)

class PssPsutilCollector(AbstractMemoryCollector):
    """使用 psutil 收集 PSS, USS, 和 RSS 内存指标。"""

    PSUTIL_METRIC_FIELDS = ["PSS_KB", "USS_KB", "RSS_KB"]

    def __init__(self, process_pattern: str, monitoring_interval: int, **kwargs):
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        self._collecting = False
        self._stop_event = False # 用于优雅停止循环
        try:
            # 编译正则表达式以提高效率
            self.compiled_pattern = re.compile(process_pattern)
        except re.error as e:
            logger.error(f"PssPsutilCollector 的正则表达式模式无效: '{process_pattern}'. 错误: {e}")
            raise ValueError(f"无效的正则表达式模式: {process_pattern}") from e


    def get_metric_fields(self) -> List[str]:
        return self.PSUTIL_METRIC_FIELDS

    def start(self):
        logger.info(f"启动 PssPsutilCollector (模式: '{self.process_pattern}', 间隔: {self.monitoring_interval}s).")
        self._collecting = True
        self._stop_event = False

    def stop(self):
        logger.info("正在停止 PssPsutilCollector。")
        self._collecting = False
        self._stop_event = True # 设置停止事件

    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        if not self._collecting:
            logger.warning("PssPsutilCollector 尚未启动。请在 read_samples() 前调用 start()。")
            return

        while not self._stop_event: # 检查停止事件
            current_interval_samples: List[ProcessMemorySample] = []
            # 在尝试采样开始时记录时间戳
            # 这有助于确保即使进程迭代花费一些时间，采样间隔也大致正确
            interval_start_time = time.monotonic()

            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'create_time']):
                if self._stop_event: # 在迭代过程中检查停止事件
                    logger.debug("PssPsutilCollector 在样本收集中检测到停止事件。")
                    break
                try:
                    # 基本检查：进程名或命令行的任何部分匹配模式
                    # cmdline() 可能返回 None 或空列表，确保处理
                    cmdline_str = " ".join(proc.info['cmdline'] or [])
                    proc_name = proc.info['name'] or "" # name() 可能返回空字符串

                    # 跳过内核进程 (通常没有命令行，并且在 memory_full_info 上可能导致 AccessDenied)
                    # 这是一个启发式方法，可能需要根据系统进行调整
                    if not cmdline_str and proc.info['pid'] > 0:
                        try:
                            if proc.info['username'] == 'root': # 许多内核线程以root身份运行
                                # 检查 /proc/[pid]/stat 的第9个字段 (flags) 是否表示内核线程
                                # 0x00200000 (PF_KTHREAD)
                                with open(f"/proc/{proc.info['pid']}/stat", 'r') as f_stat:
                                    stat_parts = f_stat.read().split()
                                    if len(stat_parts) > 8 and (int(stat_parts[8]) & 0x00200000):
                                        continue
                        except FileNotFoundError: # 进程可能刚刚结束
                            continue
                        except Exception as stat_e:
                            logger.debug(f"读取 PID {proc.info['pid']} 的 stat 文件时出错: {stat_e}")
                            # 如果无法读取stat，谨慎处理，可能不是内核线程

                    # 检查进程名或命令行是否匹配编译后的正则表达式
                    if not (self.compiled_pattern.search(proc_name) or \
                            (cmdline_str and self.compiled_pattern.search(cmdline_str))):
                        continue

                    # 获取详细内存信息 (包括 PSS, USS) 和基本内存信息 (包括 RSS)
                    # memory_full_info() 可能需要更高的权限
                    mem_full_info = proc.memory_full_info()
                    mem_info = proc.memory_info() # RSS

                    # psutil 以字节为单位返回内存，转换为 KB
                    # 检查属性是否存在，因为某些系统或旧版psutil可能没有所有字段
                    pss_kb = mem_full_info.pss / 1024 if hasattr(mem_full_info, 'pss') else 0
                    uss_kb = mem_full_info.uss / 1024 if hasattr(mem_full_info, 'uss') else 0
                    rss_kb = mem_info.rss / 1024 # rss 字段通常存在

                    metrics = {
                        "PSS_KB": int(pss_kb),
                        "USS_KB": int(uss_kb),
                        "RSS_KB": int(rss_kb),
                    }

                    sample = ProcessMemorySample(
                        pid=str(proc.info['pid']),
                        command_name=proc_name,
                        full_command=cmdline_str,
                        metrics=metrics
                    )
                    current_interval_samples.append(sample)

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    # 进程可能已终止或访问被拒绝，这在监控动态进程时是正常的
                    logger.debug(f"处理 PID {proc.pid if hasattr(proc, 'pid') else '?'} 时发生错误: {type(e).__name__} - {e}")
                    continue
                except Exception as e:
                    # 记录特定进程的其他意外错误
                    pid_val = proc.pid if hasattr(proc, 'pid') else '?'
                    name_val = proc.info.get('name', '?') if hasattr(proc, 'info') and proc.info else '?'
                    logger.warning(f"处理 PID {pid_val} ({name_val}) 时发生未知错误: {e}", exc_info=False) # 设置 exc_info=False 避免过多日志
                    continue
            
            if self._stop_event and not current_interval_samples: # 如果在收集循环中停止且没有样本，则退出
                 logger.debug("PssPsutilCollector 停止，没有样本可以生成。")
                 break

            yield current_interval_samples # 即使是空列表也生成，表示该间隔内没有匹配的进程
            
            if self._stop_event: # 再次检查，如果在yield之后立即停止
                logger.debug("PssPsutilCollector 在 yield 后检测到停止事件。")
                break

            # 计算自上次采样以来的已用时间，并休眠剩余时间
            elapsed_time = time.monotonic() - interval_start_time
            sleep_time = self.monitoring_interval - elapsed_time
            if sleep_time > 0:
                # 使睡眠可中断，以便更快地响应 stop()
                # 将长时间睡眠分成小块
                chunk_sleep = 0.1 # 100ms
                while sleep_time > 0 and not self._stop_event:
                    actual_sleep = min(sleep_time, chunk_sleep)
                    time.sleep(actual_sleep)
                    sleep_time -= actual_sleep
            elif sleep_time < 0:
                logger.warning(f"PssPsutilCollector 采样花费时间 ({elapsed_time:.2f}s) 超过监控间隔 ({self.monitoring_interval}s)。")

        logger.info("PssPsutilCollector 样本读取循环已完成。")