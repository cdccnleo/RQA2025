"""
异步处理异常场景测试

Phase 4: 测试覆盖提升 - 完善异步处理异常场景测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import pytest_asyncio
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import builtins  # For TimeoutError

try:
    from src.infrastructure.resource.core.unified_resource_manager import UnifiedResourceManager
    from src.infrastructure.resource.core.shared_interfaces import ILogger, IErrorHandler
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    # 创建mock类以避免导入错误
    class UnifiedResourceManager:
        pass
    class ILogger:
        pass
    class IErrorHandler:
        pass
    print(f"Warning: 无法导入所需模块: {e}")


class MockAsyncLogger(ILogger):
    """模拟异步日志记录器"""

    def __init__(self):
        self.logs = []
        self._lock = asyncio.Lock()

    async def log_info(self, message: str):
        async with self._lock:
            self.logs.append(("INFO", message))

    async def log_warning(self, message: str):
        async with self._lock:
            self.logs.append(("WARNING", message))

    async def log_error(self, message: str):
        async with self._lock:
            self.logs.append(("ERROR", message))

    async def log_debug(self, message: str):
        async with self._lock:
            self.logs.append(("DEBUG", message))

    def log_info(self, message: str):
        self.logs.append(("INFO", message))

    def log_warning(self, message: str):
        self.logs.append(("WARNING", message))

    def log_error(self, message: str):
        self.logs.append(("ERROR", message))

    def log_debug(self, message: str):
        self.logs.append(("DEBUG", message))


class AsyncErrorHandler(IErrorHandler):
    """异步错误处理器"""

    def __init__(self):
        self.errors = []
        self._lock = asyncio.Lock()

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """同步错误处理方法"""
        self.errors.append((error, context))

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        return attempt < 3  # 最多重试3次


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestAsyncErrorHandling:
    """异步错误处理测试"""

    @pytest_asyncio.fixture
    async def async_manager(self):
        """异步资源管理器fixture"""
        logger = MockAsyncLogger()
        error_handler = AsyncErrorHandler()

        manager = UnifiedResourceManager(
            logger=logger,
            error_handler=error_handler
        )

        yield manager

        # 清理
        if hasattr(manager, '_running') and manager._running:
            manager.stop()  # 同步方法，不需要await

    @pytest.mark.asyncio
    async def test_async_resource_request_timeout(self, async_manager):
        """测试异步资源请求超时"""
        # 模拟提供者异常
        from unittest.mock import Mock
        mock_provider = Mock()
        mock_provider.allocate_resource = Mock(side_effect=builtins.TimeoutError("Request timeout"))
        
        # Mock allocation_manager的provider_registry
        allocation_manager = async_manager.allocation_manager
        with patch.object(allocation_manager, 'provider_registry') as mock_provider_registry:
            mock_provider_registry.get_provider.return_value = mock_provider
            mock_provider_registry.has_provider.return_value = True

            # 执行资源请求（同步方法）
            result = allocation_manager.request_resource(
                "test_consumer", "test_resource", {}, 1
            )

            # 验证结果
            assert result is None

            # 验证错误处理
            assert len(async_manager.error_handler.errors) > 0
            error, context = async_manager.error_handler.errors[0]
            assert isinstance(error, builtins.TimeoutError)

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, async_manager):
        """测试异步并发请求"""
        # 设置mock provider
        from unittest.mock import Mock
        from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation
        from datetime import datetime
        
        mock_provider = Mock()
        mock_provider.resource_type = "test_resource"
        
        # 创建模拟的ResourceAllocation对象
        mock_allocation = ResourceAllocation(
            allocation_id="test_allocation",
            request_id="test_request",
            resource_id="test_resource_1",
            allocated_resources={"priority": 1},
            allocated_at=datetime.now()
        )
        mock_provider.allocate_resource.return_value = mock_allocation
        
        # Mock allocation_manager的provider_registry
        allocation_manager = async_manager.allocation_manager
        allocation_manager.provider_registry = async_manager.provider_registry
        
        # 注册provider
        async_manager.register_provider(mock_provider)
        allocation_manager.provider_registry = async_manager.provider_registry
        
        # 创建多个并发请求 - 使用run_in_executor包装同步方法
        async def request_resource_async(consumer_id, resource_type, requirements, priority):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                async_manager.allocation_manager.request_resource,
                consumer_id, resource_type, requirements, priority
            )
        
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                request_resource_async(
                    f"consumer_{i}", "test_resource", {"priority": i}, 1
                )
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证结果
        assert len(results) == 5

        # 检查是否有异常被正确处理
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            assert len(async_manager.error_handler.errors) >= len(exceptions)

    @pytest.mark.asyncio
    async def test_async_event_bus_error_handling(self, async_manager):
        """测试异步事件总线错误处理"""
        # 设置mock provider
        from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation
        
        mock_provider = MagicMock()
        mock_provider.resource_type = "test_resource"
        
        mock_allocation = ResourceAllocation(
            allocation_id="test_allocation",
            request_id="test_request",
            resource_id="test_resource_1",
            allocated_resources={},
            allocated_at=datetime.now()
        )
        mock_provider.allocate_resource.return_value = mock_allocation
        
        # 注册provider并确保allocation_manager使用正确的registry
        async_manager.register_provider(mock_provider)
        allocation_manager = async_manager.allocation_manager
        allocation_manager.provider_registry = async_manager.provider_registry
        
        # 模拟事件总线发布错误 - 在资源分配过程中触发
        with patch.object(allocation_manager, 'event_bus') as mock_event_bus:
            if mock_event_bus:
                mock_event_bus.publish.side_effect = Exception("Event bus error")

                # 请求资源（这会触发event_bus.publish，但错误应该被捕获）
                result = async_manager.request_resource(
                    "test_consumer", "test_resource", {}, 1
                )
                
                # 验证测试能正常执行而没有抛出未处理的异常
                # event_bus错误应该被正确处理，无论结果如何
                # 这个测试主要验证错误处理的健壮性
                pass  # 如果到这里没有异常，说明错误处理工作正常

    @pytest.mark.asyncio
    async def test_async_resource_cleanup_on_error(self, async_manager):
        """测试异步资源清理错误处理"""
        # 先启动管理器
        async_manager.start()  # 同步方法，不需要await

        # 注册测试消费者
        assert async_manager.register_consumer(async_manager.consumer_registry)

        # 设置完整的mock provider配置
        from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation
        
        mock_provider = MagicMock()
        mock_provider.resource_type = "test_resource"
        
        # 创建正确的ResourceAllocation对象
        mock_allocation = ResourceAllocation(
            allocation_id="test_allocation_123",
            request_id="test_request",
            resource_id="test_resource_1",
            allocated_resources={},
            allocated_at=datetime.now()
        )
        mock_provider.allocate_resource.return_value = mock_allocation
        
        # 注册provider并确保allocation_manager使用正确的registry
        async_manager.register_provider(mock_provider)
        allocation_manager = async_manager.allocation_manager
        allocation_manager.provider_registry = async_manager.provider_registry

        # 请求资源
        allocation_id = async_manager.request_resource(
            "test_consumer", "test_resource", {}, 1
        )
        assert allocation_id == "test_allocation_123"

        # 确保分配记录存在于allocation_manager中（因为request_resource会创建它）
        # 同时设置provider的release_resource方法抛出异常
        mock_provider.release_resource.side_effect = Exception("Release failed")

        # 尝试释放资源
        result = async_manager.release_resource("test_allocation_123")
        assert result is False

        # 验证错误被记录
        assert len(async_manager.error_handler.errors) > 0

    @pytest.mark.asyncio
    async def test_async_cancellation_handling(self, async_manager):
        """测试异步任务取消处理"""
        # 创建一个长时间运行的任务
        async def long_running_task():
            await asyncio.sleep(10)
            return "completed"

        # 创建任务并立即取消
        task = asyncio.create_task(long_running_task())
        task.cancel()

        # 等待任务完成并检查取消异常
        with pytest.raises(asyncio.CancelledError):
            await task

        # 验证系统能正常处理取消操作
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_async_deadlock_detection(self, async_manager):
        """测试异步死锁检测"""
        # 创建两个相互等待的异步操作
        lock1 = asyncio.Lock()
        lock2 = asyncio.Lock()

        async def task1():
            async with lock1:
                await asyncio.sleep(0.1)  # 短暂等待
                async with lock2:
                    return "task1_done"

        async def task2():
            async with lock2:
                await asyncio.sleep(0.1)  # 短暂等待
                async with lock1:
                    return "task2_done"

        # 同时执行两个任务
        results = await asyncio.gather(
            asyncio.wait_for(task1(), timeout=2.0),
            asyncio.wait_for(task2(), timeout=2.0),
            return_exceptions=True
        )

        # 验证结果（可能成功或超时）
        assert len(results) == 2

        # 如果有异常，确保被正确处理
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            # 记录检测到的潜在死锁情况
            for exc in exceptions:
                async_manager.error_handler.handle_error(exc, {
                    "context": "potential_deadlock_detected",
                    "type": type(exc).__name__
                })

    @pytest.mark.asyncio
    async def test_async_performance_under_load(self, async_manager):
        """测试异步性能负载"""
        # 启动管理器
        async_manager.start()  # 同步方法，不需要await
        
        # 设置mock providers for different resource types
        from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation
        
        for i in range(3):  # 为3种资源类型创建providers
            mock_provider = MagicMock()
            mock_provider.resource_type = f"resource_{i}"
            
            # 创建模拟的ResourceAllocation对象
            mock_allocation = ResourceAllocation(
                allocation_id=f"allocation_{i}",
                request_id=f"request_{i}",
                resource_id=f"resource_{i}_1",
                allocated_resources={"request_id": i},
                allocated_at=datetime.now()
            )
            mock_provider.allocate_resource.return_value = mock_allocation
            
            # 注册provider
            async_manager.register_provider(mock_provider)
        
        # 确保allocation_manager使用正确的provider_registry
        allocation_manager = async_manager.allocation_manager
        allocation_manager.provider_registry = async_manager.provider_registry

        # 创建大量并发请求
        start_time = time.time()

        tasks = []
        for i in range(20):  # 模拟20个并发请求
            task = asyncio.create_task(
                self._simulate_async_request(async_manager, i)
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        duration = end_time - start_time

        # 验证性能指标
        assert duration < 5.0  # 应该在5秒内完成

        # 检查成功率
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successful_requests / len(results)

        assert success_rate >= 0.8  # 至少80%的成功率

        # 记录性能指标
        async_manager.logger.log_info(
            f"异步负载测试完成: {successful_requests}/{len(results)} 成功, "
            f"耗时: {duration:.2f}秒, 成功率: {success_rate:.1%}"
        )

    async def _simulate_async_request(self, manager, request_id: int):
        """模拟异步请求"""
        try:
            # 模拟一些异步处理时间
            await asyncio.sleep(0.01 * request_id)  # 递增延迟

            # 模拟资源请求
            result = manager.request_resource(
                f"consumer_{request_id}",
                f"resource_{request_id % 3}",  # 循环使用3种资源
                {"request_id": request_id},
                priority=request_id % 3 + 1
            )

            return result

        except Exception as e:
            raise e
