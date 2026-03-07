#!/usr/bin/env python3
"""
样本文件清理测试

测试样本文件清理功能
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow


class TestSampleCleanup:
    """样本文件清理测试类"""
    
    @pytest.mark.asyncio
    async def test_cleanup_old_samples(self):
        """测试清理旧样本文件"""
        orchestrator = DataCollectionWorkflow(config={
            "sample_cleanup": {
                "max_samples_per_source": 3,
                "cleanup_on_generate": True,
                "async_cleanup": False
            }
        })
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_dir = Path(temp_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建多个测试文件
            source_id = "test_source"
            for i in range(5):
                csv_file = samples_dir / f"{source_id}_股票_{int(time.time()) + i}.csv"
                json_file = samples_dir / f"{source_id}_股票_{int(time.time()) + i}.json"
                csv_file.write_text("test data")
                json_file.write_text("{}")
                time.sleep(0.01)  # 确保文件时间不同
            
            # 执行清理
            await orchestrator._cleanup_old_samples(source_id, samples_dir)
            
            # 验证只保留了最新的3个文件
            remaining_files = list(samples_dir.glob(f"{source_id}_*.csv"))
            assert len(remaining_files) == 3
    
    @pytest.mark.asyncio
    async def test_cleanup_preserves_paired_files(self):
        """测试清理时保留配对的CSV和JSON文件"""
        orchestrator = DataCollectionWorkflow(config={
            "sample_cleanup": {
                "max_samples_per_source": 2,
                "cleanup_on_generate": True,
                "async_cleanup": False
            }
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_dir = Path(temp_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            source_id = "test_source"
            # 创建配对的CSV和JSON文件
            for i in range(3):
                timestamp = int(time.time()) + i
                csv_file = samples_dir / f"{source_id}_股票_{timestamp}.csv"
                json_file = samples_dir / f"{source_id}_股票_{timestamp}.json"
                csv_file.write_text("test data")
                json_file.write_text("{}")
                time.sleep(0.01)
            
            # 执行清理
            await orchestrator._cleanup_old_samples(source_id, samples_dir)
            
            # 验证CSV和JSON文件数量一致
            csv_files = list(samples_dir.glob(f"{source_id}_*.csv"))
            json_files = list(samples_dir.glob(f"{source_id}_*.json"))
            assert len(csv_files) == 2
            assert len(json_files) == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_no_files_to_delete(self):
        """测试无需清理的情况"""
        orchestrator = DataCollectionWorkflow(config={
            "sample_cleanup": {
                "max_samples_per_source": 10,
                "cleanup_on_generate": True,
                "async_cleanup": False
            }
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_dir = Path(temp_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            source_id = "test_source"
            # 只创建3个文件，少于限制
            for i in range(3):
                csv_file = samples_dir / f"{source_id}_股票_{int(time.time()) + i}.csv"
                csv_file.write_text("test data")
                time.sleep(0.01)
            
            # 执行清理
            await orchestrator._cleanup_old_samples(source_id, samples_dir)
            
            # 验证所有文件都保留
            remaining_files = list(samples_dir.glob(f"{source_id}_*.csv"))
            assert len(remaining_files) == 3
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_errors_gracefully(self):
        """测试清理时错误处理"""
        orchestrator = DataCollectionWorkflow(config={
            "sample_cleanup": {
                "max_samples_per_source": 2,
                "cleanup_on_generate": True,
                "async_cleanup": False
            }
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_dir = Path(temp_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            source_id = "test_source"
            # 创建文件
            csv_file = samples_dir / f"{source_id}_股票_{int(time.time())}.csv"
            csv_file.write_text("test data")
            
            # 模拟文件删除失败（通过删除文件后再尝试清理）
            csv_file.unlink()
            
            # 执行清理应该不会抛出异常
            await orchestrator._cleanup_old_samples(source_id, samples_dir)
            
            # 验证没有异常抛出
