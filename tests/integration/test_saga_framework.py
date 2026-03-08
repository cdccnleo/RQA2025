"""
Saga框架集成测试

测试Saga分布式事务框架的核心功能，包括：
1. Saga编排器功能测试
2. 补偿事务测试
3. 与数据管道集成测试
4. 与交易系统集成测试
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# 导入Saga框架组件
from src.infrastructure.saga_framework import (
    SagaOrchestrator,
    SagaDefinition,
    SagaStep,
    SagaContext,
    SagaStatus
)


class TestSagaOrchestrator:
    """Saga编排器集成测试"""
    
    @pytest.fixture
    def orchestrator(self):
        """创建Saga编排器实例"""
        return SagaOrchestrator(max_concurrent=10)
        
    @pytest.fixture
    def sample_saga_definition(self):
        """创建示例Saga定义"""
        
        async def step1_action(context: SagaContext):
            """步骤1：数据准备"""
            return {"data_prepared": True, "data_id": "12345"}
            
        async def step1_compensation(context: SagaContext):
            """步骤1补偿"""
            print("Compensating step1: Cleaning up data")
            
        async def step2_action(context: SagaContext):
            """步骤2：数据处理"""
            return {"data_processed": True, "result": "success"}
            
        async def step2_compensation(context: SagaContext):
            """步骤2补偿"""
            print("Compensating step2: Reverting processing")
            
        async def step3_action(context: SagaContext):
            """步骤3：数据存储"""
            return {"data_stored": True, "storage_id": "store_123"}
            
        async def step3_compensation(context: SagaContext):
            """步骤3补偿"""
            print("Compensating step3: Removing stored data")
            
        return SagaDefinition(
            name="test_data_processing",
            steps=[
                SagaStep(
                    name="data_preparation",
                    action=step1_action,
                    compensation=step1_compensation
                ),
                SagaStep(
                    name="data_processing",
                    action=step2_action,
                    compensation=step2_compensation
                ),
                SagaStep(
                    name="data_storage",
                    action=step3_action,
                    compensation=step3_compensation
                )
            ]
        )
        
    @pytest.mark.asyncio
    async def test_saga_success_flow(self, orchestrator, sample_saga_definition):
        """测试Saga成功流程"""
        # 注册Saga定义
        orchestrator.register_saga(sample_saga_definition)
        
        # 创建上下文
        context = SagaContext(
            saga_id="test_saga_001",
            data={"test_data": "value"}
        )
        
        # 启动Saga
        instance = await orchestrator.start_saga(
            "test_data_processing",
            context
        )
        
        # 验证结果
        assert instance.status == SagaStatus.COMPLETED
        assert len(instance.completed_steps) == 3
        assert "data_preparation" in instance.completed_steps
        assert "data_processing" in instance.completed_steps
        assert "data_storage" in instance.completed_steps
        
    @pytest.mark.asyncio
    async def test_saga_compensation_flow(self, orchestrator):
        """测试Saga补偿流程"""
        
        async def failing_step(context: SagaContext):
            """会失败的步骤"""
            raise Exception("Simulated failure")
            
        async def compensation(context: SagaContext):
            """补偿操作"""
            context.set("compensated", True)
            
        # 创建会失败的Saga定义
        failing_saga = SagaDefinition(
            name="failing_saga",
            steps=[
                SagaStep(
                    name="step1",
                    action=lambda ctx: {"step1": "completed"},
                    compensation=compensation
                ),
                SagaStep(
                    name="step2",
                    action=failing_step,
                    compensation=compensation
                )
            ]
        )
        
        orchestrator.register_saga(failing_saga)
        
        context = SagaContext(saga_id="test_saga_002")
        instance = await orchestrator.start_saga("failing_saga", context)
        
        # 验证补偿执行
        assert instance.status == SagaStatus.COMPENSATED
        assert "step1" in instance.compensation_steps
        
    @pytest.mark.asyncio
    async def test_saga_timeout(self, orchestrator, sample_saga_definition):
        """测试Saga超时处理"""
        
        async def slow_step(context: SagaContext):
            await asyncio.sleep(2)
            return {"result": "slow"}
            
        slow_saga = SagaDefinition(
            name="slow_saga",
            steps=[
                SagaStep(name="slow_step", action=slow_step)
            ]
        )
        
        orchestrator.register_saga(slow_saga)
        
        context = SagaContext(saga_id="test_saga_003")
        
        # 设置1秒超时
        instance = await orchestrator.start_saga(
            "slow_saga",
            context,
            timeout=1.0
        )
        
        # 验证超时失败
        assert instance.status == SagaStatus.FAILED
        
    @pytest.mark.asyncio
    async def test_concurrent_sagas(self, orchestrator, sample_saga_definition):
        """测试并发Saga执行"""
        orchestrator.register_saga(sample_saga_definition)
        
        # 启动5个并发Saga
        tasks = []
        for i in range(5):
            context = SagaContext(saga_id=f"concurrent_saga_{i}")
            task = orchestrator.start_saga("test_data_processing", context)
            tasks.append(task)
            
        instances = await asyncio.gather(*tasks)
        
        # 验证所有Saga完成
        for instance in instances:
            assert instance.status == SagaStatus.COMPLETED
            
    @pytest.mark.asyncio
    async def test_saga_context_propagation(self, orchestrator):
        """测试Saga上下文传递"""
        
        async def step_with_context(context: SagaContext):
            # 验证上下文数据
            assert context.get("initial_data") == "test_value"
            # 添加新数据
            return {"step_data": "step_result"}
            
        saga = SagaDefinition(
            name="context_test",
            steps=[
                SagaStep(name="step1", action=step_with_context)
            ]
        )
        
        orchestrator.register_saga(saga)
        
        context = SagaContext(
            saga_id="context_test_001",
            data={"initial_data": "test_value"}
        )
        
        instance = await orchestrator.start_saga("context_test", context)
        
        assert instance.status == SagaStatus.COMPLETED
        assert instance.context.get("step_data") == "step_result"


class TestSagaWithDataPipeline:
    """Saga与数据管道集成测试"""
    
    @pytest.mark.asyncio
    async def test_strategy_deployment_saga(self):
        """测试策略部署Saga流程"""
        
        from src.infrastructure.saga_framework import SagaOrchestrator, SagaDefinition, SagaStep
        
        orchestrator = SagaOrchestrator()
        
        # 模拟策略部署的各个步骤
        async def data_preparation(context: SagaContext):
            """数据准备"""
            print("Preparing training data...")
            return {"data_version": "v1.0", "samples": 10000}
            
        async def feature_engineering(context: SagaContext):
            """特征工程"""
            print("Engineering features...")
            return {"features": 50, "feature_version": "v2.0"}
            
        async def model_training(context: SagaContext):
            """模型训练"""
            print("Training model...")
            return {"model_id": "model_123", "accuracy": 0.85}
            
        async def model_validation(context: SagaContext):
            """模型验证"""
            print("Validating model...")
            return {"validation_passed": True, "sharpe": 1.5}
            
        async def deployment(context: SagaContext):
            """部署"""
            print("Deploying model...")
            return {"deployment_id": "deploy_456", "status": "active"}
            
        strategy_deployment_saga = SagaDefinition(
            name="strategy_deployment",
            steps=[
                SagaStep(name="data_preparation", action=data_preparation),
                SagaStep(name="feature_engineering", action=feature_engineering),
                SagaStep(name="model_training", action=model_training),
                SagaStep(name="model_validation", action=model_validation),
                SagaStep(name="deployment", action=deployment)
            ]
        )
        
        orchestrator.register_saga(strategy_deployment_saga)
        
        context = SagaContext(
            saga_id="strategy_deploy_001",
            data={"strategy_name": "TestStrategy"}
        )
        
        instance = await orchestrator.start_saga("strategy_deployment", context)
        
        assert instance.status == SagaStatus.COMPLETED
        assert len(instance.completed_steps) == 5
        assert instance.context.get("model_id") == "model_123"
        assert instance.context.get("deployment_id") == "deploy_456"


class TestSagaWithTradingSystem:
    """Saga与交易系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_order_execution_saga(self):
        """测试订单执行Saga流程"""
        
        from src.infrastructure.saga_framework import SagaOrchestrator, SagaDefinition, SagaStep
        
        orchestrator = SagaOrchestrator()
        
        # 模拟交易执行的各个步骤
        async def validate_order(context: SagaContext):
            """验证订单"""
            order = context.get("order")
            assert order is not None
            print(f"Validating order: {order}")
            return {"validation_passed": True}
            
        async def check_position(context: SagaContext):
            """检查持仓"""
            print("Checking position limits...")
            return {"position_checked": True, "available": 1000}
            
        async def reserve_funds(context: SagaContext):
            """预留资金"""
            print("Reserving funds...")
            return {"funds_reserved": True, "reserved_amount": 50000}
            
        async def submit_order(context: SagaContext):
            """提交订单"""
            print("Submitting order to exchange...")
            return {"order_submitted": True, "exchange_order_id": "EX12345"}
            
        async def confirm_execution(context: SagaContext):
            """确认成交"""
            print("Confirming execution...")
            return {"execution_confirmed": True, "filled_qty": 100}
            
        order_execution_saga = SagaDefinition(
            name="order_execution",
            steps=[
                SagaStep(name="validate_order", action=validate_order),
                SagaStep(name="check_position", action=check_position),
                SagaStep(name="reserve_funds", action=reserve_funds),
                SagaStep(name="submit_order", action=submit_order),
                SagaStep(name="confirm_execution", action=confirm_execution)
            ]
        )
        
        orchestrator.register_saga(order_execution_saga)
        
        context = SagaContext(
            saga_id="order_exec_001",
            data={
                "order": {
                    "symbol": "000001.SZ",
                    "quantity": 100,
                    "price": 10.5,
                    "side": "buy"
                }
            }
        )
        
        instance = await orchestrator.start_saga("order_execution", context)
        
        assert instance.status == SagaStatus.COMPLETED
        assert instance.context.get("exchange_order_id") == "EX12345"
        assert instance.context.get("filled_qty") == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
