"""
特征提取任务持久化测试
验证任务创建、保存、加载和更新的完整流程
"""

import pytest
import os
import json
import time
from pathlib import Path
from datetime import datetime

# 导入持久化模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.gateway.web.feature_task_persistence import (
    save_feature_task,
    load_feature_task,
    list_feature_tasks,
    update_feature_task,
    delete_feature_task,
    ensure_directories
)


class TestFeatureTaskPersistence:
    """特征任务持久化测试"""
    
    def setup_method(self):
        """测试前准备"""
        ensure_directories()
        # 清理测试数据
        test_task_id = "test_task_123"
        delete_feature_task(test_task_id)
    
    def test_save_and_load_task(self):
        """测试保存和加载任务"""
        task = {
            "task_id": "test_task_123",
            "task_type": "技术指标",
            "status": "pending",
            "progress": 0,
            "feature_count": 0,
            "start_time": int(time.time()),
            "config": {
                "indicators": ["MA", "RSI"],
                "symbols": ["000001"]
            }
        }
        
        # 保存任务
        assert save_feature_task(task) == True
        
        # 加载任务
        loaded_task = load_feature_task("test_task_123")
        assert loaded_task is not None
        assert loaded_task["task_id"] == "test_task_123"
        assert loaded_task["task_type"] == "技术指标"
        assert loaded_task["status"] == "pending"
        assert "saved_at" in loaded_task
    
    def test_list_tasks(self):
        """测试列出任务"""
        # 创建多个测试任务
        for i in range(3):
            task = {
                "task_id": f"test_task_{i}",
                "task_type": "技术指标",
                "status": "running" if i % 2 == 0 else "completed",
                "progress": i * 25,
                "feature_count": i * 5,
                "start_time": int(time.time()) - i * 100,
                "config": {}
            }
            save_feature_task(task)
        
        # 列出所有任务
        all_tasks = list_feature_tasks(limit=10)
        assert len(all_tasks) >= 3
        
        # 列出运行中的任务
        running_tasks = list_feature_tasks(status="running", limit=10)
        assert len(running_tasks) >= 1
        assert all(t["status"] == "running" for t in running_tasks)
    
    def test_update_task(self):
        """测试更新任务"""
        # 创建任务
        task = {
            "task_id": "test_task_update",
            "task_type": "技术指标",
            "status": "pending",
            "progress": 0,
            "feature_count": 0,
            "start_time": int(time.time()),
            "config": {}
        }
        save_feature_task(task)
        
        # 更新任务
        updates = {
            "status": "running",
            "progress": 50,
            "feature_count": 10
        }
        assert update_feature_task("test_task_update", updates) == True
        
        # 验证更新
        updated_task = load_feature_task("test_task_update")
        assert updated_task["status"] == "running"
        assert updated_task["progress"] == 50
        assert updated_task["feature_count"] == 10
        assert "updated_at" in updated_task
    
    def test_delete_task(self):
        """测试删除任务"""
        # 创建任务
        task = {
            "task_id": "test_task_delete",
            "task_type": "技术指标",
            "status": "completed",
            "progress": 100,
            "feature_count": 20,
            "start_time": int(time.time()),
            "config": {}
        }
        save_feature_task(task)
        
        # 验证任务存在
        assert load_feature_task("test_task_delete") is not None
        
        # 删除任务
        assert delete_feature_task("test_task_delete") == True
        
        # 验证任务已删除
        assert load_feature_task("test_task_delete") is None
    
    def test_file_system_persistence(self):
        """测试文件系统持久化"""
        task = {
            "task_id": "test_file_task",
            "task_type": "统计特征",
            "status": "running",
            "progress": 30,
            "feature_count": 5,
            "start_time": int(time.time()),
            "config": {"test": True}
        }
        
        save_feature_task(task)
        
        # 验证文件存在
        data_dir = Path(__file__).parent.parent.parent / "data" / "feature_tasks"
        file_path = data_dir / "test_file_task.json"
        assert file_path.exists()
        
        # 验证文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        assert file_data["task_id"] == "test_file_task"
        assert file_data["task_type"] == "统计特征"
        assert "saved_at" in file_data
    
    def test_task_creation_integration(self):
        """测试任务创建集成"""
        from src.gateway.web.feature_engineering_service import create_feature_task
        
        # 创建任务
        task = create_feature_task(
            task_type="技术指标",
            config={
                "indicators": ["MA", "RSI", "MACD"],
                "symbols": ["000001", "000002"]
            }
        )
        
        assert task is not None
        assert "task_id" in task
        assert task["task_type"] == "技术指标"
        assert task["status"] == "pending"
        
        # 验证任务已持久化
        loaded_task = load_feature_task(task["task_id"])
        assert loaded_task is not None
        assert loaded_task["task_id"] == task["task_id"]
        assert loaded_task["config"]["indicators"] == ["MA", "RSI", "MACD"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

