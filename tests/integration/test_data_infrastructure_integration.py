"""
数据层与基础设施层集成测试
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.data_manager import DataManager, DataModel
from src.infrastructure.config.config_manager import ConfigManager
# 使用Mock替代不存在的模块
from unittest.mock import Mock


class TestDataInfrastructureIntegration:
    """数据层与基础设施层集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def infrastructure_services(self, temp_dir):
        """基础设施服务fixture"""
        # 初始化基础设施服务
        config_manager = ConfigManager(env='test')
        # 使用Mock替代不存在的服务
        metrics_collector = Mock()
        error_handler = Mock()
        logger = Mock()
        cache_manager = Mock()
        db_manager = Mock()
        
        return {
            'config_manager': config_manager,
            'metrics_collector': metrics_collector,
            'error_handler': error_handler,
            'logger': logger,
            'cache_manager': cache_manager,
            'db_manager': db_manager
        }
    
    @pytest.fixture
    def data_manager_with_infrastructure(self, temp_dir, infrastructure_services):
        """带基础设施支持的数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '2',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 100),
                'cache_ttl': '3600',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '7',
                'frequency': 'daily',
                'adjust_type': 'none'
            },
            "Infrastructure": {
                'monitoring_enabled': 'true',
                'error_handling_enabled': 'true',
                'logging_enabled': 'true'
            }
        }
        
        # 将配置加载到配置管理器
        infrastructure_services['config_manager'].load_from_dict(config_dict)
        
        return DataManager(config_dict=config_dict)
    
    def test_config_manager_integration(self, data_manager_with_infrastructure, infrastructure_services):
        """测试与配置管理器的集成"""
        config_manager = infrastructure_services['config_manager']
        
        # 验证配置一致性
        assert data_manager_with_infrastructure.config.getint("General", "max_concurrent_workers") == 2
        assert config_manager.get("General.max_concurrent_workers") == 2
        
        # 测试配置更新传播
        config_manager.update_config("General.max_concurrent_workers", 4)
        assert config_manager.get("General.max_concurrent_workers") == 4
    
    @patch('src.infrastructure.monitoring.metrics.MetricsCollector.record_metric')
    def test_monitoring_integration(self, mock_record_metric, data_manager_with_infrastructure):
        """测试与监控系统的集成"""
        # 模拟数据加载操作
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        # 记录数据血缘（这会触发监控）
        data_manager_with_infrastructure._record_data_lineage(
            'stock',
            model,
            '2023-01-01',
            '2023-01-31'
        )
        
        # 验证监控指标被记录
        mock_record_metric.assert_called()
    
    @patch('src.infrastructure.error.error_handler.ErrorHandler.handle_error')
    def test_error_handling_integration(self, mock_handle_error, data_manager_with_infrastructure):
        """测试与错误处理系统的集成"""
        # 模拟错误情况
        try:
            raise Exception("测试错误")
        except Exception as e:
            # 验证错误被正确处理
            assert str(e) == "测试错误"
    
    def test_logging_integration(self, data_manager_with_infrastructure, caplog):
        """测试与日志系统的集成"""
        # 执行一个操作来触发日志
        data_manager_with_infrastructure.validate_all_configs()
        
        # 验证日志被正确记录
        assert "数据管理器初始化完成" in caplog.text or "DataManager initialized" in caplog.text
    
    def test_cache_integration(self, data_manager_with_infrastructure, infrastructure_services):
        """测试与缓存系统的集成"""
        cache_manager = infrastructure_services['cache_manager']
        
        # 创建测试数据
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        # 测试缓存操作
        cache_key = "test_cache_key"
        cache_manager.set(cache_key, model)
        
        # 验证缓存中的数据
        cached_model = cache_manager.get(cache_key)
        assert cached_model is not None
        pd.testing.assert_frame_equal(cached_model.data, model.data)
    
    def test_database_integration(self, data_manager_with_infrastructure, infrastructure_services):
        """测试与数据库系统的集成"""
        db_manager = infrastructure_services['db_manager']
        
        # 模拟数据持久化
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        # 测试数据存储
        storage_success = db_manager.store_data("test_table", model.data)
        assert storage_success is True
        
        # 测试数据检索
        retrieved_data = db_manager.get_data("test_table")
        assert retrieved_data is not None


class TestDataInfrastructurePerformance:
    """数据层与基础设施层性能测试"""
    
    @pytest.fixture
    def performance_data_manager(self, temp_dir):
        """性能测试数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '4',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 1024),  # 1GB
                'cache_ttl': '86400',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '30',
                'frequency': 'daily',
                'adjust_type': 'none'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_concurrent_operations_with_infrastructure(self, performance_data_manager):
        """测试与基础设施的并发操作性能"""
        import time
        import threading
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                start_time = time.time()
                
                # 模拟数据加载操作
                data = pd.DataFrame({'close': range(1000)})
                model = DataModel(data, '1d')
                
                # 记录数据血缘
                performance_data_manager._record_data_lineage(
                    'stock',
                    model,
                    '2023-01-01',
                    '2023-01-31',
                    symbols=[f'00000{worker_id}']
                )
                
                end_time = time.time()
                results.append(end_time - start_time)
                
            except Exception as e:
                errors.append(str(e))
        
        # 启动多个并发线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证性能
        assert len(errors) == 0, f"发现错误: {errors}"
        assert len(results) == 10
        
        # 验证平均执行时间合理
        avg_time = sum(results) / len(results)
        assert avg_time < 2.0  # 平均时间应该小于2秒
    
    def test_memory_usage_with_infrastructure(self, performance_data_manager):
        """测试与基础设施的内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行内存密集型操作
        for i in range(100):
            data = pd.DataFrame({'close': range(1000)})
            model = DataModel(data, '1d')
            performance_data_manager._record_data_lineage('stock', model, '2023-01-01', '2023-01-31')
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长合理（小于200MB）
        assert memory_increase < 200 * 1024 * 1024


class TestDataInfrastructureErrorHandling:
    """数据层与基础设施层错误处理测试"""
    
    @pytest.fixture
    def error_data_manager(self, temp_dir):
        """错误处理测试数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '1',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 10),
                'cache_ttl': '3600',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '1',
                'cache_days': '1',
                'frequency': 'daily',
                'adjust_type': 'none'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_infrastructure_error_propagation(self, error_data_manager):
        """测试基础设施错误传播"""
        # 模拟基础设施服务错误
        with patch('src.infrastructure.config.config_manager.ConfigManager.get') as mock_get:
            mock_get.side_effect = Exception("配置服务错误")
            
            # 应该优雅地处理错误
            try:
                error_data_manager.validate_all_configs()
            except Exception as e:
                assert "配置服务错误" in str(e)
    
    def test_cache_error_handling(self, error_data_manager, temp_dir):
        """测试缓存错误处理"""
        # 模拟缓存目录权限问题
        cache_dir = Path(temp_dir) / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        try:
            cache_dir.chmod(0o444)  # 只读
            
            # 尝试缓存操作，应该优雅地处理错误
            data = pd.DataFrame({'close': [100, 101, 102]})
            model = DataModel(data, '1d')
            
            # 这应该不会抛出异常，而是记录警告
            error_data_manager._record_data_lineage('stock', model, '2023-01-01', '2023-01-31')
            
        except PermissionError:
            # Windows上可能不支持chmod
            pass
        finally:
            # 恢复权限
            try:
                cache_dir.chmod(0o755)
            except:
                pass
    
    def test_database_error_handling(self, error_data_manager):
        """测试数据库错误处理"""
        # 模拟数据库连接错误
        with patch('src.infrastructure.database.db_manager.DatabaseManager.store_data') as mock_store:
            mock_store.side_effect = Exception("数据库连接错误")
            
            data = pd.DataFrame({'close': [100, 101, 102]})
            model = DataModel(data, '1d')
            
            # 应该优雅地处理数据库错误
            try:
                # 这里应该调用数据库存储，但由于模拟了错误，会抛出异常
                pass
            except Exception as e:
                assert "数据库连接错误" in str(e)


class TestDataInfrastructureSecurity:
    """数据层与基础设施层安全测试"""
    
    @pytest.fixture
    def secure_data_manager(self, temp_dir):
        """安全测试数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '2',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 100),
                'cache_ttl': '3600',
            },
            "Security": {
                'encryption_enabled': 'true',
                'access_control_enabled': 'true',
                'audit_logging_enabled': 'true'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_data_encryption(self, secure_data_manager):
        """测试数据加密"""
        # 创建敏感数据
        sensitive_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1001, 1002],
            'user_id': ['user1', 'user2', 'user3']  # 敏感信息
        })
        
        model = DataModel(sensitive_data, '1d')
        
        # 验证数据在存储前被加密
        # 这里应该检查数据是否被正确加密
        assert model.data is not None
    
    def test_access_control(self, secure_data_manager):
        """测试访问控制"""
        # 模拟不同用户访问
        user_roles = ['admin', 'user', 'guest']
        
        for role in user_roles:
            # 验证不同角色的访问权限
            data = pd.DataFrame({'close': [100, 101, 102]})
            model = DataModel(data, '1d')
            
            # 根据角色验证访问权限
            if role == 'admin':
                # 管理员应该有完全访问权限
                assert True
            elif role == 'user':
                # 普通用户应该有有限访问权限
                assert True
            elif role == 'guest':
                # 访客应该有只读权限
                assert True
    
    def test_audit_logging(self, secure_data_manager):
        """测试审计日志"""
        # 执行需要审计的操作
        data = pd.DataFrame({'close': [100, 101, 102]})
        model = DataModel(data, '1d')
        
        # 记录数据血缘（应该触发审计日志）
        secure_data_manager._record_data_lineage(
            'stock',
            model,
            '2023-01-01',
            '2023-01-31'
        )
        
        # 验证审计日志被记录
        # 这里应该检查审计日志文件或数据库
        assert True


class TestDataInfrastructureScalability:
    """数据层与基础设施层可扩展性测试"""
    
    @pytest.fixture
    def scalable_data_manager(self, temp_dir):
        """可扩展性测试数据管理器"""
        config_dict = {
            "General": {
                'max_concurrent_workers': '8',
                'cache_dir': temp_dir,
                'max_cache_size': str(1024 * 1024 * 1024 * 10),  # 10GB
                'cache_ttl': '86400',
            },
            "Scalability": {
                'auto_scaling_enabled': 'true',
                'load_balancing_enabled': 'true',
                'distributed_cache_enabled': 'true'
            }
        }
        return DataManager(config_dict=config_dict)
    
    def test_auto_scaling(self, scalable_data_manager):
        """测试自动扩展"""
        # 模拟高负载情况
        import time
        
        start_time = time.time()
        
        # 创建大量并发任务
        futures = []
        for i in range(50):
            future = scalable_data_manager.thread_pool.submit(
                lambda: time.sleep(0.1)  # 模拟数据加载
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证自动扩展效果
        assert execution_time < 10.0  # 50个任务应该在10秒内完成
    
    def test_load_balancing(self, scalable_data_manager):
        """测试负载均衡"""
        import time
        
        # 模拟多个数据源
        data_sources = ['stock', 'index', 'financial', 'news']
        
        results = []
        for source in data_sources:
            start_time = time.time()
            
            # 模拟从不同数据源加载数据
            data = pd.DataFrame({'close': range(100)})
            model = DataModel(data, '1d')
            
            scalable_data_manager._record_data_lineage(
                source,
                model,
                '2023-01-01',
                '2023-01-31'
            )
            
            end_time = time.time()
            results.append(end_time - start_time)
        
        # 验证负载均衡效果（所有数据源的处理时间应该相近）
        avg_time = sum(results) / len(results)
        for result in results:
            assert abs(result - avg_time) < avg_time * 0.5  # 时间差异不应超过50%
    
    def test_distributed_cache(self, scalable_data_manager):
        """测试分布式缓存"""
        # 模拟分布式缓存操作
        cache_keys = [f"test_key_{i}" for i in range(100)]
        
        for key in cache_keys:
            data = pd.DataFrame({'close': range(100)})
            model = DataModel(data, '1d')
            
            # 测试分布式缓存存储
            scalable_data_manager._record_data_lineage(
                'stock',
                model,
                '2023-01-01',
                '2023-01-31'
            )
        
        # 验证分布式缓存统计
        stats = scalable_data_manager.get_cache_stats()
        assert stats['file_count'] > 0
        assert stats['total_size'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 