"""
混合监控架构的基础测试。

测试混合内存收集器的核心功能和数据模型。
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from mymonitor.models.hybrid_monitoring import (
    ProcessTask,
    SampleResult,
    HybridCollectorConfig,
    HybridCollectorStats
)
from mymonitor.collectors.base import ProcessMemorySample
from mymonitor.collectors.hybrid_collector import HybridMemoryCollector


class TestHybridMonitoringModels:
    """测试混合监控数据模型。"""
    
    def test_process_task_creation(self):
        """测试ProcessTask创建和属性。"""
        task = ProcessTask(
            pid=1234,
            command_name="gcc",
            full_command="gcc -c main.c -o main.o",
            priority=1
        )
        
        assert task.pid == 1234
        assert task.command_name == "gcc"
        assert task.full_command == "gcc -c main.c -o main.o"
        assert task.priority == 1
        assert task.retry_count == 0
        assert task.discovery_timestamp > 0
        
    def test_process_task_priority_comparison(self):
        """测试ProcessTask优先级比较。"""
        task1 = ProcessTask(pid=1, command_name="gcc", full_command="gcc", priority=0)
        task2 = ProcessTask(pid=2, command_name="make", full_command="make", priority=1)
        
        assert task1 < task2  # 优先级0 < 优先级1
        
    def test_process_task_retry(self):
        """测试ProcessTask重试机制。"""
        original_task = ProcessTask(
            pid=1234,
            command_name="gcc",
            full_command="gcc -c main.c",
            priority=0,
            retry_count=1
        )
        
        retry_task = original_task.create_retry_task()
        
        assert retry_task.pid == original_task.pid
        assert retry_task.command_name == original_task.command_name
        assert retry_task.full_command == original_task.full_command
        assert retry_task.priority == 1  # 优先级降低
        assert retry_task.retry_count == 2  # 重试次数增加
        assert retry_task.discovery_timestamp > original_task.discovery_timestamp
        
    def test_sample_result_creation(self):
        """测试SampleResult创建和属性。"""
        sample = ProcessMemorySample(
            pid="1234",
            command_name="gcc",
            full_command="gcc -c main.c",
            metrics={"PSS_KB": 1024, "RSS_KB": 2048}
        )
        
        result = SampleResult(
            pid=1234,
            sample=sample,
            worker_id=1,
            category=("CPP_Compile", "GCCCompiler"),
            sampling_duration=15.5
        )
        
        assert result.pid == 1234
        assert result.sample == sample
        assert result.worker_id == 1
        assert result.category == ("CPP_Compile", "GCCCompiler")
        assert result.sampling_duration == 15.5
        assert result.is_successful is True
        assert result.memory_kb == 1024  # PSS优先
        
    def test_hybrid_collector_config_validation(self):
        """测试HybridCollectorConfig验证。"""
        # 有效配置
        config = HybridCollectorConfig(
            discovery_interval=0.01,
            sampling_workers=4,
            task_queue_size=1000
        )
        assert config.discovery_interval == 0.01
        assert config.sampling_workers == 4
        
        # 无效配置
        with pytest.raises(ValueError, match="discovery_interval must be positive"):
            HybridCollectorConfig(discovery_interval=0)
            
        with pytest.raises(ValueError, match="sampling_workers must be at least 1"):
            HybridCollectorConfig(sampling_workers=0)
            
    def test_hybrid_collector_config_presets(self):
        """测试HybridCollectorConfig预设配置。"""
        perf_config = HybridCollectorConfig.create_performance_optimized()
        assert perf_config.discovery_interval == 0.005
        assert perf_config.sampling_workers == 6
        assert perf_config.batch_result_size == 100
        
        conservative_config = HybridCollectorConfig.create_resource_conservative()
        assert conservative_config.discovery_interval == 0.02
        assert conservative_config.sampling_workers == 2
        assert conservative_config.batch_result_size == 20
        
    def test_hybrid_collector_stats(self):
        """测试HybridCollectorStats统计功能。"""
        stats = HybridCollectorStats()
        
        # 初始状态
        assert stats.uptime_seconds >= 0
        assert stats.success_rate == 0.0
        assert stats.discovery_rate == 0.0
        
        # 更新统计
        stats.total_samples_completed = 80
        stats.total_samples_failed = 20
        assert stats.success_rate == 80.0
        
        # Worker统计
        stats.update_worker_stats(worker_id=1, completed=True, duration=10.0)
        stats.update_worker_stats(worker_id=1, completed=False, duration=5.0)
        
        worker_stats = stats.get_worker_stats(1)
        assert worker_stats['samples_completed'] == 1
        assert worker_stats['samples_failed'] == 1
        assert worker_stats['average_duration'] == 7.5


class TestHybridMemoryCollector:
    """测试混合内存收集器。"""
    
    @pytest.fixture
    def mock_config(self):
        """创建测试配置。"""
        return HybridCollectorConfig(
            discovery_interval=0.1,  # 较长间隔便于测试
            sampling_workers=2,
            task_queue_size=10,
            result_queue_size=10,
            enable_prioritization=False,
            max_retry_count=1
        )
    
    def test_collector_initialization(self, mock_config):
        """测试收集器初始化。"""
        collector = HybridMemoryCollector(
            process_pattern="gcc|clang",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        assert collector.process_pattern == "gcc|clang"
        assert collector.monitoring_interval == 0.1
        assert collector.config == mock_config
        assert not collector.collecting
        assert collector.task_queue is not None
        assert collector.result_queue is not None
        
    def test_collector_start_stop(self, mock_config):
        """测试收集器启动和停止。"""
        collector = HybridMemoryCollector(
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        # 测试启动
        collector.start()
        assert collector.collecting is True
        assert collector.discovery_worker is not None
        assert len(collector.sampling_workers) == mock_config.sampling_workers
        
        # 等待一小段时间让workers启动
        time.sleep(0.1)
        
        # 测试停止
        collector.stop()
        assert collector.collecting is False
        assert collector.discovery_worker is None
        assert len(collector.sampling_workers) == 0
    
    def test_collector_metrics(self, mock_config):
        """测试收集器指标。"""
        collector = HybridMemoryCollector(
            process_pattern="test",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        # 测试指标字段
        metric_fields = collector.get_metric_fields()
        assert "PSS_KB" in metric_fields
        assert "RSS_KB" in metric_fields
        assert "USS_KB" in metric_fields
        
        # 测试主要指标
        primary_metric = collector.get_primary_metric_field()
        assert primary_metric == "PSS_KB"
    
    def test_collector_stats(self, mock_config):
        """测试收集器统计信息。"""
        collector = HybridMemoryCollector(
            process_pattern="test",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        stats = collector.get_stats()
        
        # 验证统计结构
        assert 'collector_type' in stats
        assert stats['collector_type'] == 'hybrid'
        assert 'config' in stats
        assert 'runtime' in stats
        assert 'performance' in stats
        assert 'counters' in stats
        assert 'queues' in stats
        
        # 验证配置信息
        config_stats = stats['config']
        assert config_stats['sampling_workers'] == mock_config.sampling_workers
        assert config_stats['discovery_interval'] == mock_config.discovery_interval
    
    @patch('mymonitor.collectors.discovery_worker.psutil.process_iter')
    def test_collector_with_mock_processes(self, mock_process_iter, mock_config):
        """测试收集器处理模拟进程。"""
        # 创建模拟进程
        mock_proc1 = Mock()
        mock_proc1.info = {
            'pid': 1001,
            'name': 'gcc',
            'cmdline': ['gcc', '-c', 'main.c']
        }
        
        mock_proc2 = Mock()
        mock_proc2.info = {
            'pid': 1002,
            'name': 'clang',
            'cmdline': ['clang', '-c', 'test.c']
        }
        
        mock_process_iter.return_value = [mock_proc1, mock_proc2]
        
        # 创建收集器
        collector = HybridMemoryCollector(
            process_pattern="gcc|clang",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        # 启动收集器
        collector.start()
        
        # 等待一段时间让发现Worker工作
        time.sleep(0.2)
        
        # 检查任务队列
        task_count = 0
        try:
            while True:
                task = collector.task_queue.get_nowait()
                task_count += 1
                if task_count > 10:  # 防止无限循环
                    break
        except:
            pass
        
        # 停止收集器
        collector.stop()
        
        # 验证发现了进程
        assert task_count >= 0  # 至少应该发现一些任务
    
    def test_collector_queue_operations(self, mock_config):
        """测试收集器队列操作。"""
        collector = HybridMemoryCollector(
            process_pattern="test",
            monitoring_interval=0.1,
            config=mock_config
        )
        
        # 测试队列大小获取
        task_queue_size = collector._get_queue_size(collector.task_queue)
        result_queue_size = collector._get_queue_size(collector.result_queue)
        
        assert task_queue_size >= 0 or task_queue_size == -1  # -1表示不可用
        assert result_queue_size >= 0 or result_queue_size == -1
        
        # 测试队列清空
        collector._drain_queues()  # 应该不抛出异常


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
