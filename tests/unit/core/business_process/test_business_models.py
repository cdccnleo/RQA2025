"""
测试核心业务层业务模型功能
"""
import pytest
from datetime import datetime

# 尝试导入所需模块
try:
    from core.business_process.models.models import (
        BusinessModel, TradingBusinessModel, RiskBusinessModel, ModelType, BusinessModelManager
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBusinessModel:
    """测试业务模型基类"""

    def test_business_model_initialization(self):
        """测试业务模型初始化"""
        model = BusinessModel(
            model_id="test_model_001",
            model_name="Test Model",
            model_type="trading",
            description="Test description",
            config={"param1": "value1"}
        )

        assert model.model_type == "trading"
        assert model.model_name == "Test Model"
        assert model.description == "Test description"
        assert model.config == {"param1": "value1"}
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)

    def test_business_model_validation(self):
        """测试业务模型验证"""
        # 有效模型
        model = BusinessModel(
            model_id="test_model_001",
            model_name="Valid Model",
            model_type="trading",
            description="Valid description"
        )
        assert model.validate()

        # 无效模型 - 空名称
        model = BusinessModel(
            model_id="test_model_002",
            model_name="",
            model_type="trading",
            description="Valid description"
        )
        assert not model.validate()

    def test_business_model_metadata_operations(self):
        """测试模型元数据操作"""
        model = BusinessModel(
            model_id="test_model_003",
            model_name="Test Model",
            model_type="trading",
            description="Test description"
        )

        # 测试元数据操作
        assert model.get_metadata("test_key") is None
        assert model.get_metadata("test_key", "default") == "default"

        model.update_metadata("status", "active")
        assert model.get_metadata("status") == "active"

        # 测试配置更新
        model.config["learning_rate"] = 0.01
        assert model.config["learning_rate"] == 0.01


class TestTradingBusinessModel:
    """测试交易业务模型"""

    def test_trading_model_initialization(self):
        """测试交易模型初始化"""
        model = TradingBusinessModel(
            model_id="trading_model_001",
            model_name="Trading Strategy Model",
            model_type="trading",
            description="A trading strategy model",
            trading_strategy="mean_reversion",
            config={"lookback": 20, "threshold": 0.01}
        )

        assert model.model_type == "trading"
        assert model.trading_strategy == "mean_reversion"
        assert model.config["lookback"] == 20
        assert model.config["threshold"] == 0.01

    def test_trading_model_execution(self):
        """测试交易模型执行"""
        model = TradingBusinessModel(
            model_id="trading_model_002",
            model_name="Test Trading Model",
            model_type="trading",
            description="Test",
            trading_strategy="momentum"
        )

        # 模拟市场数据
        market_data = {
            "price": 100.0,
            "volume": 1000,
            "timestamp": datetime.now()
        }

        # 验证交易模型
        assert model.validate_trading_model()

        # 更新交易指标
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": 0.05}
        model.update_trading_metrics(metrics)
        assert model.performance_metrics["sharpe_ratio"] == 1.5
        assert model.performance_metrics["max_drawdown"] == 0.05


class TestRiskBusinessModel:
    """测试风险业务模型"""

    def test_risk_model_initialization(self):
        """测试风险模型初始化"""
        model = RiskBusinessModel(
            model_id="risk_model_001",
            model_name="Risk Assessment Model",
            model_type="risk",
            description="A risk assessment model",
            risk_strategy="portfolio_risk"
        )

        assert model.model_type == "risk"
        assert model.risk_strategy == "portfolio_risk"

    def test_risk_model_assessment(self):
        """测试风险模型评估"""
        model = RiskBusinessModel(
            model_id="risk_model_002",
            model_name="Test Risk Model",
            model_type="risk",
            description="Test",
            risk_strategy="value_at_risk"
        )

        # 模拟投资组合数据
        portfolio_data = {
            "positions": {"AAPL": 100, "GOOGL": 50},
            "prices": {"AAPL": 150.0, "GOOGL": 2500.0},
            "returns": [0.01, -0.005, 0.008, 0.002]
        }

        # 验证风险模型
        assert model.validate_risk_model()

        # 更新风险阈值
        thresholds = {"var_limit": 0.05, "stress_test_threshold": 0.1}
        model.risk_thresholds.update(thresholds)
        assert model.risk_thresholds["var_limit"] == 0.05


class TestModelIntegration:
    """测试模型集成"""

    def test_model_type_consistency(self):
        """测试模型类型一致性"""
        trading_model = TradingBusinessModel(
            model_id="trading_001",
            model_name="Trading Model",
            model_type="trading"
        )
        risk_model = RiskBusinessModel(
            model_id="risk_001",
            model_name="Risk Model",
            model_type="risk"
        )

        assert trading_model.model_type == "trading"
        assert risk_model.model_type == "risk"

    def test_model_validation_integration(self):
        """测试模型验证集成"""
        # 有效的交易模型
        trading_model = TradingBusinessModel(
            model_id="trading_002",
            model_name="Valid Trading Model",
            model_type="trading",
            trading_strategy="momentum"
        )
        assert trading_model.validate()
        assert trading_model.validate_trading_model()

        # 有效的风险模型
        risk_model = RiskBusinessModel(
            model_id="risk_002",
            model_name="Valid Risk Model",
            model_type="risk",
            risk_strategy="portfolio_risk"
        )
        assert risk_model.validate()

    def test_get_model(self):
        """测试获取模型"""
        manager = BusinessModelManager()
        model = TradingBusinessModel(
            model_id="trading_model_004",
            model_name="Test Model",
            model_type="trading",
            description="Test"
        )

        manager.register_model("test_model", model)
        retrieved_model = manager.get_model("test_model")
        assert retrieved_model == model

        # 获取不存在的模型
        assert manager.get_model("nonexistent") is None

    def test_model_lifecycle_management(self):
        """测试模型生命周期管理"""
        manager = BusinessModelManager()

        # 注册模型
        model = TradingBusinessModel(
            model_id="trading_model_004",
            model_name="Test Model",
            model_type="trading",
            description="Test"
        )
        manager.register_model("test_model", model)

        # 验证模型注册成功
        retrieved = manager.get_model("test_model")
        assert retrieved == model
        assert retrieved.model_name == "Test Model"

    def test_list_models_by_type(self):
        """测试按类型列出模型"""
        manager = BusinessModelManager()

        # 注册不同类型的模型
        trading_model = TradingBusinessModel(
            model_id="trading_model_005",
            model_name="Trading Model",
            model_type="trading",
            description="Trading model"
        )
        risk_model = RiskBusinessModel(
            model_id="risk_model_003",
            model_name="Risk Model",
            model_type="risk",
            description="Risk model"
        )

        manager.register_model("trading", trading_model)
        manager.register_model("risk", risk_model)

        # 验证可以获取注册的模型
        retrieved_trading = manager.get_model("trading")
        retrieved_risk = manager.get_model("risk")
        assert retrieved_trading == trading_model
        assert retrieved_risk == risk_model

        # 列出所有模型
        all_models = manager.list_models()
        assert len(all_models) == 2
        assert "trading" in all_models
        assert "risk" in all_models

    def test_model_performance_monitoring(self):
        """测试模型性能监控"""
        manager = BusinessModelManager()
        model = TradingBusinessModel(
            model_id="trading_model_004",
            model_name="Test Model",
            model_type="trading",
            description="Test"
        )

        success = manager.register_model("test_model", model)
        assert success

        # 验证模型注册成功
        retrieved = manager.get_model("test_model")
        assert retrieved == model

        # 验证模型可以更新指标
        metrics = {"sharpe_ratio": 1.5, "win_rate": 0.65}
        retrieved.update_trading_metrics(metrics)
        assert retrieved.performance_metrics["sharpe_ratio"] == 1.5
        assert retrieved.performance_metrics["win_rate"] == 0.65
