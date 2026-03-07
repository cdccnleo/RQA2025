#!/usr/bin/env python3
"""
核心功能验证脚本
验证RQA2025系统的核心模块是否正常工作
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_imports() -> Tuple[bool, List[str]]:
    """测试核心模块导入"""
    logger = logging.getLogger(__name__)
    errors = []
    success = True

    modules_to_test = [
        ("src.trading.trading_engine", "TradingEngine"),
        ("src.trading.order_manager", "OrderManager"),
        ("src.trading.live_trader", "LiveTrader"),
        ("src.trading.backtester", "Backtester"),
        ("src.trading.risk", "风险模块"),
        ("src.models", "模型模块"),
        ("src.data", "数据模块"),
        ("src.features", "特征模块"),
        ("src.utils", "工具模块"),
    ]

    logger.info("开始测试模块导入...")

    for module_path, module_name in modules_to_test:
        try:
            __import__(module_path)
            logger.info(f"✓ {module_name} 导入成功")
        except ImportError as e:
            logger.error(f"✗ {module_name} 导入失败: {e}")
            errors.append(f"{module_name}: {e}")
            success = False
        except Exception as e:
            logger.error(f"✗ {module_name} 导入异常: {e}")
            errors.append(f"{module_name}: {e}")
            success = False

    return success, errors


def test_trading_engine() -> Tuple[bool, List[str]]:
    """测试交易引擎核心功能"""
    logger = logging.getLogger(__name__)
    errors = []
    success = True

    try:
        from src.trading.trading_engine import OrderType, OrderDirection, OrderStatus

        logger.info("✓ 交易引擎枚举类导入成功")

        # 测试枚举值
        assert OrderType.MARKET.value == 1
        assert OrderDirection.BUY.value == 1
        assert OrderStatus.PENDING.value == 1

        logger.info("✓ 交易引擎枚举值验证成功")

    except Exception as e:
        logger.error(f"✗ 交易引擎测试失败: {e}")
        errors.append(f"交易引擎: {e}")
        success = False

    return success, errors


def test_china_market_adapter() -> Tuple[bool, List[str]]:
    """测试A股市场适配器"""
    logger = logging.getLogger(__name__)
    errors = []
    success = True

    try:
        from src.trading.trading_engine import ChinaMarketAdapter

        # 测试ST股票检查
        result = ChinaMarketAdapter.check_trade_restrictions("ST000001", 10.0, 10.0)
        assert result == False, "ST股票应该被限制交易"

        # 测试正常股票
        result = ChinaMarketAdapter.check_trade_restrictions("000001", 10.0, 10.0)
        assert result == True, "正常股票应该可以交易"

        logger.info("✓ A股市场适配器测试成功")

    except Exception as e:
        logger.error(f"✗ A股市场适配器测试失败: {e}")
        errors.append(f"A股市场适配器: {e}")
        success = False

    return success, errors


def test_data_structures() -> Tuple[bool, List[str]]:
    """测试数据结构"""
    logger = logging.getLogger(__name__)
    errors = []
    success = True

    try:
        import pandas as pd
        import numpy as np

        # 测试pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3, "DataFrame长度应该为3"

        # 测试numpy
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0, "数组平均值应该为3.0"

        logger.info("✓ 数据结构测试成功")

    except Exception as e:
        logger.error(f"✗ 数据结构测试失败: {e}")
        errors.append(f"数据结构: {e}")
        success = False

    return success, errors


def test_config_files() -> Tuple[bool, List[str]]:
    """测试配置文件"""
    logger = logging.getLogger(__name__)
    errors = []
    success = True

    config_files = [
        "pyproject.toml",
        "pytest.ini",
        "requirements-clean.txt",
        "requirements-dev.txt",
    ]

    for config_file in config_files:
        file_path = project_root / config_file
        if file_path.exists():
            logger.info(f"✓ 配置文件存在: {config_file}")
        else:
            logger.warning(f"⚠ 配置文件缺失: {config_file}")
            errors.append(f"配置文件缺失: {config_file}")
            success = False

    return success, errors


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始RQA2025核心功能验证...")

    all_errors = []
    all_success = True

    # 测试模块导入
    success, errors = test_imports()
    all_success &= success
    all_errors.extend(errors)

    # 测试交易引擎
    success, errors = test_trading_engine()
    all_success &= success
    all_errors.extend(errors)

    # 测试A股市场适配器
    success, errors = test_china_market_adapter()
    all_success &= success
    all_errors.extend(errors)

    # 测试数据结构
    success, errors = test_data_structures()
    all_success &= success
    all_errors.extend(errors)

    # 测试配置文件
    success, errors = test_config_files()
    all_success &= success
    all_errors.extend(errors)

    # 输出结果
    logger.info("\n" + "="*50)
    if all_success:
        logger.info("🎉 所有核心功能验证通过!")
    else:
        logger.error("❌ 部分功能验证失败:")
        for error in all_errors:
            logger.error(f"  - {error}")

    logger.info("="*50)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
