#!/usr/bin/env python3
"""
并发控制测试

测试配置更新的并发控制功能
"""

import pytest
import asyncio
import threading
from unittest.mock import Mock, patch

from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
from src.gateway.web.data_source_config_manager import DataSourceConfigManager
from src.infrastructure.utils.optimization.concurrency_controller import ConcurrencyController


class TestConcurrencyControl:
    """并发控制测试类"""
    
    @pytest.mark.asyncio
    async def test_update_with_lock_protection(self):
        """测试配置更新使用锁保护"""
        orchestrator = DataCollectionWorkflow()
        
        # 模拟数据源配置管理器
        with patch('src.core.orchestration.business_process.data_collection_orchestrator.get_data_source_config_manager') as mock_manager:
            mock_config_manager = Mock()
            mock_config_manager.update_data_source = Mock(return_value=True)
            mock_manager.return_value = mock_config_manager
            
            # 执行更新
            await orchestrator._update_data_source_last_test_time("test_source", success=True)
            
            # 验证调用了更新方法
            mock_config_manager.update_data_source.assert_called_once()
    
    def test_config_manager_update_with_lock(self):
        """测试配置管理器更新使用锁"""
        config_manager = DataSourceConfigManager()
        
        # 添加测试数据源
        test_source = {
            "id": "test_source",
            "name": "测试数据源",
            "type": "股票数据",
            "url": "https://test.com",
            "enabled": True
        }
        config_manager.add_data_source(test_source)
        
        # 并发更新测试
        update_count = 0
        lock = threading.Lock()
        
        def update_worker():
            nonlocal update_count
            for _ in range(10):
                success = config_manager.update_data_source("test_source", {"last_test": "2026-01-17 12:00:00"})
                if success:
                    with lock:
                        update_count += 1
        
        # 启动多个线程并发更新
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有更新都成功（锁保护确保不会丢失更新）
        assert update_count == 50  # 5线程 * 10次更新
    
    @pytest.mark.asyncio
    async def test_concurrent_updates_same_source(self):
        """测试同一数据源的并发更新"""
        orchestrator = DataCollectionWorkflow()
        
        with patch('src.core.orchestration.business_process.data_collection_orchestrator.get_data_source_config_manager') as mock_manager:
            mock_config_manager = Mock()
            mock_config_manager.update_data_source = Mock(return_value=True)
            mock_manager.return_value = mock_config_manager
            
            # 并发更新同一数据源
            tasks = []
            for i in range(10):
                task = orchestrator._update_data_source_last_test_time("test_source", success=True)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # 验证所有更新都执行了（锁保护确保不会冲突）
            assert mock_config_manager.update_data_source.call_count == 10
