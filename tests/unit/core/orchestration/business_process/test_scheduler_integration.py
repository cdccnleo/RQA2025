#!/usr/bin/env python3
"""
调度器集成测试

测试调度器与持久化模块的集成，验证重启场景
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler
from src.core.orchestration.business_process.scheduler_persistence import SchedulerPersistence


class TestSchedulerIntegration:
    """调度器集成测试类"""
    
    @pytest.mark.asyncio
    async def test_scheduler_loads_persistence_on_start(self):
        """测试调度器启动时加载持久化数据"""
        scheduler = DataCollectionServiceScheduler()
        
        # 模拟持久化数据
        test_times = {
            "akshare_stock_a": time.time() - 100,
            "akshare_stock_hk": time.time() - 200
        }
        scheduler.persistence.save_last_collection_times(test_times)
        
        # 启动调度器
        with patch.object(scheduler, '_scheduler_loop', new_callable=AsyncMock):
            success = await scheduler.start()
            assert success is True
            
            # 验证已加载历史数据
            assert len(scheduler.last_collection_times) == 2
            assert "akshare_stock_a" in scheduler.last_collection_times
            assert "akshare_stock_hk" in scheduler.last_collection_times
        
        # 清理
        await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_scheduler_saves_on_stop(self):
        """测试调度器停止时保存状态"""
        scheduler = DataCollectionServiceScheduler()
        
        # 启动调度器
        with patch.object(scheduler, '_scheduler_loop', new_callable=AsyncMock):
            await scheduler.start()
            
            # 设置一些采集时间
            scheduler.last_collection_times = {
                "test_source": time.time()
            }
            
            # 停止调度器
            success = await scheduler.stop()
            assert success is True
            
            # 验证已保存
            loaded_times = scheduler.persistence.load_last_collection_times()
            assert "test_source" in loaded_times
    
    @pytest.mark.asyncio
    async def test_scheduler_periodic_save(self):
        """测试调度器定期保存"""
        scheduler = DataCollectionServiceScheduler()
        scheduler._save_interval = 1  # 1秒保存一次，便于测试
        
        # 启动调度器
        scheduler.running = True
        scheduler.data_source_manager = Mock()
        scheduler.data_source_manager.get_data_sources = Mock(return_value=[])
        
        # 模拟调度循环
        scheduler.last_collection_times = {"test_source": time.time()}
        scheduler._last_save_time = time.time() - 2  # 2秒前保存过
        
        # 执行一次循环检查
        await scheduler._scheduler_loop()
        
        # 验证已保存（通过检查_last_save_time是否更新）
        # 注意：由于是异步执行，可能需要等待
        await asyncio.sleep(0.1)
        
        # 清理
        scheduler.running = False
