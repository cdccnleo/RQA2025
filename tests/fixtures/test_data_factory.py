#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据工厂

提供标准化的测试数据准备和清理机制
"""

import os
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path


class TestDataFactory:
    """测试数据工厂类"""
    
    def __init__(self):
        """初始化测试数据工厂"""
        self._temp_files: List[str] = []
        self._temp_dirs: List[str] = []
        
    def create_temp_file(self, content: str = "", suffix: str = ".txt", prefix: str = "test_") -> str:
        """
        创建临时文件
        
        Args:
            content: 文件内容
            suffix: 文件后缀
            prefix: 文件前缀
            
        Returns:
            临时文件路径
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            os.close(fd)
            raise
            
        self._temp_files.append(path)
        return path
        
    def create_temp_dir(self, prefix: str = "test_") -> str:
        """
        创建临时目录
        
        Args:
            prefix: 目录前缀
            
        Returns:
            临时目录路径
        """
        path = tempfile.mkdtemp(prefix=prefix)
        self._temp_dirs.append(path)
        return path
        
    def cleanup(self):
        """清理所有创建的临时文件和目录"""
        # 清理临时文件
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"清理临时文件失败: {file_path}, 错误: {e}")
                
        # 清理临时目录
        for dir_path in self._temp_dirs:
            try:
                if os.path.exists(dir_path):
                    import shutil
                    shutil.rmtree(dir_path)
            except Exception as e:
                print(f"清理临时目录失败: {dir_path}, 错误: {e}")
                
        self._temp_files.clear()
        self._temp_dirs.clear()
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动清理"""
        self.cleanup()
        return False


class DatabaseTestData:
    """数据库测试数据生成器"""
    
    @staticmethod
    def create_user_data(count: int = 10) -> List[Dict[str, Any]]:
        """
        创建用户测试数据
        
        Args:
            count: 数据条数
            
        Returns:
            用户数据列表
        """
        return [
            {
                "id": i + 1,
                "name": f"test_user_{i}",
                "email": f"test_{i}@example.com",
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for i in range(count)
        ]
        
    @staticmethod
    def create_market_data(symbol: str = "AAPL", count: int = 100) -> List[Dict[str, Any]]:
        """
        创建市场数据
        
        Args:
            symbol: 股票代码
            count: 数据条数
            
        Returns:
            市场数据列表
        """
        base_price = 150.0
        return [
            {
                "symbol": symbol,
                "timestamp": (datetime.now() - timedelta(minutes=count-i)).isoformat(),
                "open": round(base_price + (i % 10), 2),
                "high": round(base_price + (i % 10) + 2, 2),
                "low": round(base_price + (i % 10) - 1, 2),
                "close": round(base_price + (i % 10) + 1, 2),
                "volume": 1000000 + i * 10000,
            }
            for i in range(count)
        ]
        
    @staticmethod
    def create_config_data() -> Dict[str, Any]:
        """
        创建配置测试数据
        
        Returns:
            配置数据字典
        """
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_password",
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
            },
            "influxdb": {
                "url": "http://localhost:8086",
                "token": "test_token",
                "org": "test_org",
                "bucket": "test_bucket",
            },
            "logging": {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }


class MockDataGenerator:
    """Mock数据生成器"""
    
    @staticmethod
    def create_query_result(
        success: bool = True,
        row_count: int = 10,
        data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        创建查询结果数据
        
        Args:
            success: 是否成功
            row_count: 行数
            data: 数据内容
            
        Returns:
            查询结果字典
        """
        if data is None:
            data = [{"id": i, "value": f"data_{i}"} for i in range(row_count)]
            
        return {
            "success": success,
            "row_count": row_count,
            "data": data,
            "execution_time": 0.01,
            "query_id": "test_query_001",
            "metadata": {"source": "test"},
        }
        
    @staticmethod
    def create_write_result(
        success: bool = True,
        affected_rows: int = 1,
    ) -> Dict[str, Any]:
        """
        创建写入结果数据
        
        Args:
            success: 是否成功
            affected_rows: 影响行数
            
        Returns:
            写入结果字典
        """
        return {
            "success": success,
            "affected_rows": affected_rows,
            "execution_time": 0.005,
            "write_id": "test_write_001",
        }
        
    @staticmethod
    def create_health_check_result(
        is_healthy: bool = True,
        message: str = "Healthy",
    ) -> Dict[str, Any]:
        """
        创建健康检查结果数据
        
        Args:
            is_healthy: 是否健康
            message: 消息
            
        Returns:
            健康检查结果字典
        """
        return {
            "is_healthy": is_healthy,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "database": "OK",
                "cache": "OK",
                "queue": "OK",
            },
        }


# 便捷函数
def get_test_data_factory() -> TestDataFactory:
    """获取测试数据工厂实例"""
    return TestDataFactory()


def get_database_test_data() -> DatabaseTestData:
    """获取数据库测试数据生成器"""
    return DatabaseTestData()


def get_mock_data_generator() -> MockDataGenerator:
    """获取Mock数据生成器"""
    return MockDataGenerator()

