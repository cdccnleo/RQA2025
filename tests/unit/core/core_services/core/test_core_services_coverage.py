"""
Core Services Core组件测试覆盖率补充

补充business_service、database_service、strategy_manager的测试覆盖
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import uuid

try:
    from src.core.core_services.core.business_service import (
        BusinessProcess,
        BusinessProcessStatus,
        BusinessProcessType,
        TradingStrategy,
        StrategyService,
        OrderService,
        PortfolioService,
        ProcessService,
        DataAnalysisService,
        BusinessService
    )
    from src.core.core_services.core.strategy_manager import (
        Strategy,
        StrategyManager
    )
    CORE_SERVICES_IMPORTS_AVAILABLE = True
except ImportError as e:
    CORE_SERVICES_IMPORTS_AVAILABLE = False
    pytest.skip(f"Core Services导入失败: {e}", allow_module_level=True)


@pytest.fixture
def mock_config_manager():
    """创建Mock配置管理器"""
    return Mock()


@pytest.fixture
def mock_db_service():
    """创建Mock数据库服务"""
    return Mock()


@pytest.fixture
def strategy_service(mock_config_manager, mock_db_service):
    """创建StrategyService实例"""
    return StrategyService(mock_config_manager, mock_db_service)


class TestBusinessProcess:
    """测试BusinessProcess数据类"""

    def test_business_process_creation(self):
        """测试创建业务流程"""
        process = BusinessProcess(
            process_id="test_process_1",
            process_type=BusinessProcessType.STRATEGY_EXECUTION,
            user_id=123
        )
        
        assert process.process_id == "test_process_1"
        assert process.process_type == BusinessProcessType.STRATEGY_EXECUTION
        assert process.user_id == 123
        assert process.status == BusinessProcessStatus.PENDING
        assert process.progress == 0.0

    def test_business_process_with_parameters(self):
        """测试带参数的业务流程"""
        process = BusinessProcess(
            process_id="test_process_2",
            process_type=BusinessProcessType.ORDER_PROCESSING,
            user_id=456,
            parameters={"symbol": "000001", "quantity": 100},
            status=BusinessProcessStatus.RUNNING
        )
        
        assert process.parameters == {"symbol": "000001", "quantity": 100}
        assert process.status == BusinessProcessStatus.RUNNING

    def test_business_process_status_enum(self):
        """测试业务流程状态枚举"""
        assert BusinessProcessStatus.PENDING.value == "pending"
        assert BusinessProcessStatus.RUNNING.value == "running"
        assert BusinessProcessStatus.COMPLETED.value == "completed"
        assert BusinessProcessStatus.FAILED.value == "failed"
        assert BusinessProcessStatus.CANCELLED.value == "cancelled"

    def test_business_process_type_enum(self):
        """测试业务流程类型枚举"""
        assert BusinessProcessType.STRATEGY_EXECUTION.value == "strategy_execution"
        assert BusinessProcessType.ORDER_PROCESSING.value == "order_processing"
        assert BusinessProcessType.PORTFOLIO_REBALANCE.value == "portfolio_rebalance"
        assert BusinessProcessType.RISK_ASSESSMENT.value == "risk_assessment"
        assert BusinessProcessType.DATA_PROCESSING.value == "data_processing"
        assert BusinessProcessType.MARKET_ANALYSIS.value == "market_analysis"


class TestTradingStrategy:
    """测试TradingStrategy数据类"""

    def test_trading_strategy_creation(self):
        """测试创建交易策略"""
        strategy = TradingStrategy(
            strategy_id="strategy_1",
            name="Test Strategy",
            description="Test Description",
            parameters={"param1": "value1"},
            user_id=123
        )
        
        assert strategy.strategy_id == "strategy_1"
        assert strategy.name == "Test Strategy"
        assert strategy.description == "Test Description"
        assert strategy.parameters == {"param1": "value1"}
        assert strategy.user_id == 123
        assert strategy.is_active is True


class TestStrategyService:
    """测试StrategyService组件"""

    @pytest.mark.asyncio
    async def test_create_strategy_success(self, strategy_service):
        """测试成功创建策略"""
        strategy_data = {
            "name": "Test Strategy",
            "parameters": {"param1": "value1"}
        }
        
        result = await strategy_service.create_strategy(123, strategy_data)
        
        assert result["success"] is True
        assert "strategy_id" in result
        assert result["data"]["name"] == "Test Strategy"
        assert result["data"]["user_id"] == 123

    @pytest.mark.asyncio
    async def test_create_strategy_with_default_name(self, strategy_service):
        """测试使用默认名称创建策略"""
        strategy_data = {"parameters": {}}
        
        result = await strategy_service.create_strategy(456, strategy_data)
        
        assert result["success"] is True
        assert result["data"]["name"] == "Unnamed Strategy"

    @pytest.mark.asyncio
    async def test_create_strategy_exception_handling(self, strategy_service):
        """测试创建策略异常处理"""
        # Mock uuid.uuid4抛出异常
        with patch('src.core.core_services.core.business_service.uuid.uuid4', side_effect=Exception("UUID error")):
            result = await strategy_service.create_strategy(123, {})
            
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_strategy_success(self, strategy_service):
        """测试成功执行策略"""
        # 先创建策略
        strategy_data = {"name": "Test Strategy", "parameters": {}}
        create_result = await strategy_service.create_strategy(123, strategy_data)
        strategy_id = create_result["strategy_id"]
        
        # 执行策略
        market_data = {"symbol": "000001", "price": 10.5}
        result = await strategy_service.execute_strategy(strategy_id, market_data)
        
        assert result["success"] is True
        assert "execution_id" in result

    @pytest.mark.asyncio
    async def test_execute_nonexistent_strategy(self, strategy_service):
        """测试执行不存在的策略"""
        result = await strategy_service.execute_strategy("nonexistent", {})
        
        assert result["success"] is False
        assert result["error"] == "策略不存在"

    @pytest.mark.asyncio
    async def test_get_strategy(self, strategy_service):
        """测试获取策略"""
        strategy_data = {"name": "Test Strategy", "parameters": {}}
        create_result = await strategy_service.create_strategy(123, strategy_data)
        strategy_id = create_result["strategy_id"]
        
        result = await strategy_service.get_strategy(strategy_id)
        
        assert result["success"] is True
        assert result["data"]["strategy_id"] == strategy_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_strategy(self, strategy_service):
        """测试获取不存在的策略"""
        result = await strategy_service.get_strategy("nonexistent")
        
        assert result["success"] is False
        assert result["error"] == "策略不存在"

    @pytest.mark.asyncio
    async def test_list_strategies(self, strategy_service):
        """测试列出策略"""
        # 创建多个策略
        await strategy_service.create_strategy(123, {"name": "Strategy 1"})
        await strategy_service.create_strategy(123, {"name": "Strategy 2"})
        await strategy_service.create_strategy(456, {"name": "Strategy 3"})
        
        # 列出用户123的策略
        result = await strategy_service.list_strategies(123)
        
        assert result["success"] is True
        assert len(result["strategies"]) == 2

    @pytest.mark.asyncio
    async def test_update_strategy(self, strategy_service):
        """测试更新策略"""
        strategy_data = {"name": "Original Name", "parameters": {}}
        create_result = await strategy_service.create_strategy(123, strategy_data)
        strategy_id = create_result["strategy_id"]
        
        update_data = {"name": "Updated Name", "parameters": {"new_param": "value"}}
        result = await strategy_service.update_strategy(strategy_id, update_data)
        
        assert result["success"] is True
        assert result["data"]["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_strategy(self, strategy_service):
        """测试删除策略"""
        strategy_data = {"name": "Test Strategy", "parameters": {}}
        create_result = await strategy_service.create_strategy(123, strategy_data)
        strategy_id = create_result["strategy_id"]
        
        result = await strategy_service.delete_strategy(strategy_id)
        
        assert result["success"] is True
        
        # 验证策略已删除
        get_result = await strategy_service.get_strategy(strategy_id)
        assert get_result["success"] is False


class TestOrderService:
    """测试OrderService组件"""

    @pytest.fixture
    def order_service(self, mock_config_manager, mock_db_service):
        """创建OrderService实例"""
        return OrderService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_process_order_success(self, order_service):
        """测试成功处理订单"""
        order_data = {
            "symbol": "000001",
            "quantity": 100,
            "price": 10.5,
            "order_type": "market"
        }
        
        result = await order_service.process_order(123, order_data)
        
        assert result["success"] is True
        assert "order_id" in result
        assert result["order"]["symbol"] == "000001"
        assert result["order"]["quantity"] == 100

    @pytest.mark.asyncio
    async def test_process_order_exception_handling(self, order_service):
        """测试处理订单异常处理"""
        with patch('src.core.core_services.core.business_service.uuid.uuid4', side_effect=Exception("UUID error")):
            result = await order_service.process_order(123, {})
            
            assert result["success"] is False
            assert "error" in result


class TestPortfolioService:
    """测试PortfolioService组件"""

    @pytest.fixture
    def portfolio_service(self, mock_config_manager, mock_db_service):
        """创建PortfolioService实例"""
        return PortfolioService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_rebalance_portfolio_success(self, portfolio_service):
        """测试成功再平衡投资组合"""
        target_allocation = {
            "000001": 0.3,
            "000002": 0.4,
            "000003": 0.3
        }
        
        result = await portfolio_service.rebalance_portfolio(123, target_allocation)
        
        assert result["success"] is True
        assert "portfolio_id" in result
        assert result["portfolio"]["target_allocation"] == target_allocation

    @pytest.mark.asyncio
    async def test_rebalance_portfolio_exception_handling(self, portfolio_service):
        """测试再平衡投资组合异常处理"""
        with patch('src.core.core_services.core.business_service.uuid.uuid4', side_effect=Exception("UUID error")):
            result = await portfolio_service.rebalance_portfolio(123, {})
            
            assert result["success"] is False
            assert "error" in result


class TestProcessService:
    """测试ProcessService组件"""

    @pytest.fixture
    def process_service(self, mock_config_manager, mock_db_service):
        """创建ProcessService实例"""
        return ProcessService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_get_process_status_existing(self, process_service):
        """测试获取存在的流程状态"""
        process_service.active_processes["process_1"] = {
            "process_id": "process_1",
            "user_id": 123,
            "status": "running"
        }
        
        result = await process_service.get_process_status("process_1")
        
        assert result is not None
        assert result["process_id"] == "process_1"
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_process_status_nonexistent(self, process_service):
        """测试获取不存在的流程状态"""
        result = await process_service.get_process_status("nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_process_existing(self, process_service):
        """测试取消存在的流程"""
        process_service.active_processes["process_1"] = {
            "process_id": "process_1",
            "status": "running"
        }
        
        result = await process_service.cancel_process("process_1")
        
        assert result is True
        assert process_service.active_processes["process_1"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_process_nonexistent(self, process_service):
        """测试取消不存在的流程"""
        result = await process_service.cancel_process("nonexistent")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_processes(self, process_service):
        """测试获取用户流程"""
        process_service.active_processes = {
            "process_1": {"process_id": "process_1", "user_id": 123, "status": "running"},
            "process_2": {"process_id": "process_2", "user_id": 123, "status": "completed"},
            "process_3": {"process_id": "process_3", "user_id": 456, "status": "running"}
        }
        
        result = await process_service.get_user_processes(123)
        
        assert len(result) == 2
        assert all(p["user_id"] == 123 for p in result)

    @pytest.mark.asyncio
    async def test_get_user_processes_with_status_filter(self, process_service):
        """测试按状态过滤用户流程"""
        process_service.active_processes = {
            "process_1": {"process_id": "process_1", "user_id": 123, "status": "running"},
            "process_2": {"process_id": "process_2", "user_id": 123, "status": "completed"}
        }
        
        result = await process_service.get_user_processes(123, status="running")
        
        assert len(result) == 1
        assert result[0]["status"] == "running"


class TestDataAnalysisService:
    """测试DataAnalysisService组件"""

    @pytest.fixture
    def data_analysis_service(self, mock_config_manager, mock_db_service):
        """创建DataAnalysisService实例"""
        return DataAnalysisService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_analyze_market_data_success(self, data_analysis_service):
        """测试成功分析市场数据"""
        symbols = ["000001", "000002", "000003"]
        
        result = await data_analysis_service.analyze_market_data(symbols, analysis_type="technical")
        
        assert result["success"] is True
        assert "analysis" in result
        assert result["analysis"]["symbols"] == symbols
        assert result["analysis"]["analysis_type"] == "technical"

    @pytest.mark.asyncio
    async def test_analyze_market_data_exception_handling(self, data_analysis_service):
        """测试市场数据分析异常处理"""
        with patch('src.core.core_services.core.business_service.uuid.uuid4', side_effect=Exception("UUID error")):
            result = await data_analysis_service.analyze_market_data([], "technical")
            
            assert result["success"] is False
            assert "error" in result


class TestBusinessService:
    """测试BusinessService组件"""

    @pytest.fixture
    def business_service(self):
        """创建BusinessService实例"""
        with patch('src.core.core_services.core.business_service.UnifiedConfigManager'):
            return BusinessService()

    def test_business_service_initialization(self, business_service):
        """测试业务服务初始化"""
        assert business_service.config_manager is not None
        assert hasattr(business_service, 'active_processes')
        assert isinstance(business_service.active_processes, dict)

    def test_business_service_has_sub_services(self, business_service):
        """测试业务服务包含子服务"""
        assert hasattr(business_service, 'strategy_service')
        assert hasattr(business_service, 'order_service')
        assert hasattr(business_service, 'portfolio_service')
        assert hasattr(business_service, 'process_service')
        assert hasattr(business_service, 'data_analysis_service')


class TestStrategy:
    """测试Strategy抽象类"""

    def test_strategy_abstract_class(self):
        """测试Strategy抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            Strategy()

    def test_strategy_metadata(self):
        """测试策略元数据"""
        class ConcreteStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        strategy = ConcreteStrategy()
        metadata = strategy.get_metadata()
        
        assert metadata["name"] == "Test Strategy"
        assert metadata["description"] == "Test Description"
        assert metadata["class"] == "ConcreteStrategy"

    def test_strategy_validate_input(self):
        """测试策略输入验证"""
        class ConcreteStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        strategy = ConcreteStrategy()
        assert strategy.validate_input() is True


class TestStrategyManager:
    """测试StrategyManager组件"""

    def test_strategy_manager_initialization(self):
        """测试策略管理器初始化"""
        manager = StrategyManager(name="test_manager")
        
        assert manager.name == "test_manager"
        assert len(manager._strategies) == 0
        assert manager._default_strategy is None

    def test_register_strategy(self):
        """测试注册策略"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        
        manager.register_strategy(strategy)
        
        assert "Test Strategy" in manager._strategies
        assert manager._strategies["Test Strategy"] == strategy

    def test_register_strategy_with_group(self):
        """测试注册策略到组"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        
        manager.register_strategy(strategy, group="test_group")
        
        assert "test_group" in manager._strategy_groups
        assert "Test Strategy" in manager._strategy_groups["test_group"]

    def test_get_strategy(self):
        """测试获取策略"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        manager.register_strategy(strategy)
        
        retrieved = manager.get_strategy("Test Strategy")
        
        assert retrieved == strategy

    def test_get_nonexistent_strategy(self):
        """测试获取不存在的策略"""
        manager = StrategyManager()
        
        result = manager.get_strategy("Nonexistent")
        
        assert result is None

    def test_execute_strategy(self):
        """测试执行策略"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success", "args": args, "kwargs": kwargs}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        manager.register_strategy(strategy)
        
        result = manager.execute_strategy("Test Strategy", "arg1", "arg2", key1="value1")
        
        assert result["result"] == "success"
        assert "arg1" in result["args"]
        assert result["kwargs"]["key1"] == "value1"

    def test_execute_nonexistent_strategy(self):
        """测试执行不存在的策略"""
        manager = StrategyManager()
        
        with pytest.raises(ValueError, match="策略 'Nonexistent' 未注册"):
            manager.execute_strategy("Nonexistent")

    def test_list_strategies(self):
        """测试列出所有策略"""
        class Strategy1(Strategy):
            @property
            def name(self):
                return "Strategy 1"
            
            @property
            def description(self):
                return "Description 1"
            
            def execute(self, *args, **kwargs):
                return {}
        
        class Strategy2(Strategy):
            @property
            def name(self):
                return "Strategy 2"
            
            @property
            def description(self):
                return "Description 2"
            
            def execute(self, *args, **kwargs):
                return {}
        
        manager = StrategyManager()
        manager.register_strategy(Strategy1())
        manager.register_strategy(Strategy2())
        
        strategies = manager.list_strategies()
        
        assert len(strategies) == 2
        assert "Strategy 1" in strategies
        assert "Strategy 2" in strategies

    def test_set_default_strategy(self):
        """测试设置默认策略"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        manager.register_strategy(strategy)
        
        manager.set_default_strategy("Test Strategy")
        
        assert manager._default_strategy == "Test Strategy"

    def test_execute_default_strategy(self):
        """测试执行默认策略"""
        class TestStrategy(Strategy):
            @property
            def name(self):
                return "Test Strategy"
            
            @property
            def description(self):
                return "Test Description"
            
            def execute(self, *args, **kwargs):
                return {"result": "success"}
        
        manager = StrategyManager()
        strategy = TestStrategy()
        manager.register_strategy(strategy)
        manager.set_default_strategy("Test Strategy")
        
        result = manager.execute_default_strategy(key="value")
        
        assert result["result"] == "success"

