"""
内存收集器工厂类。

该模块提供 CollectorFactory 类，用于根据配置创建不同类型的内存收集器实例。
"""

import logging
from typing import Dict, Any

from .base import AbstractMemoryCollector

logger = logging.getLogger(__name__)


class CollectorFactory:
    """
    内存收集器工厂类。

    根据配置创建不同类型的内存收集器实例。
    """

    def __init__(
        self,
        metric_type: str,
        process_pattern: str,
        monitoring_interval: float,
        pss_collector_mode: str = "full_scan",
        **kwargs,
    ):
        """
        初始化收集器工厂。

        Args:
            metric_type: 指标类型 ("pss_psutil" 或 "rss_pidstat")
            process_pattern: 进程匹配模式
            monitoring_interval: 监控间隔
            pss_collector_mode: PSS收集器模式
            **kwargs: 其他参数
        """
        self.metric_type = metric_type
        self.process_pattern = process_pattern
        self.monitoring_interval = monitoring_interval
        self.pss_collector_mode = pss_collector_mode
        self.kwargs = kwargs

        logger.info(
            f"CollectorFactory initialized: metric_type={metric_type}, "
            f"pattern='{process_pattern}', interval={monitoring_interval}s"
        )

    def create_collector(self, collector_cpu_core: int) -> AbstractMemoryCollector:
        """
        创建内存收集器实例。

        Args:
            collector_cpu_core: 收集器绑定的CPU核心

        Returns:
            内存收集器实例

        Raises:
            ValueError: 如果指标类型未知
        """
        # 通用参数
        collector_kwargs = {"collector_cpu_core": collector_cpu_core, **self.kwargs}

        if self.metric_type == "pss_psutil":
            from .pss_psutil import PssPsutilCollector

            collector_kwargs["mode"] = self.pss_collector_mode
            return PssPsutilCollector(
                self.process_pattern, self.monitoring_interval, **collector_kwargs
            )
        elif self.metric_type == "rss_pidstat":
            from .rss_pidstat import RssPidstatCollector

            return RssPidstatCollector(
                self.process_pattern, self.monitoring_interval, **collector_kwargs
            )
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
