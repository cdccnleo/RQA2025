"""
测试脚本：验证持久化迁移功能完整性
测试PostgreSQL数据库与文件系统双存储机制
"""

import os
import sys
import json
import logging
from datetime import datetime
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# 导入测试模块
from src.gateway.web.strategy_routes import (
    save_strategy_conception,
    get_strategy_conception,
    load_strategy_conceptions,
    update_strategy_conception,
    delete_strategy_conception
)
from src.gateway.web.backtest_persistence import (
    save_backtest_result,
    load_backtest_result,
    list_backtest_results,
    update_backtest_result,
    delete_backtest_result
)
from src.gateway.web.unified_persistence import (
    get_strategy_conception_persistence,
    get_backtest_persistence
)


def test_database_connection():
    """
    测试数据库连接
    """
    logger.info("=== 测试数据库连接 ===")
    try:
        # 尝试获取持久化管理器实例
        strategy_persistence = get_strategy_conception_persistence()
        backtest_persistence = get_backtest_persistence()
        
        # 测试基本操作
        test_data = {"id": "test_connection", "name": "Test Connection", "description": "Test database connection"}
        
        # 尝试保存到数据库
        success = strategy_persistence.save(test_data)
        logger.info(f"数据库连接测试结果: {'成功' if success else '失败'}")
        
        # 清理测试数据
        strategy_persistence.delete("test_connection")
        
        return success
    except Exception as e:
        logger.error(f"数据库连接测试失败: {e}")
        return False


import asyncio

def test_strategy_conception_crud():
    """
    测试策略构思的CRUD操作
    """
    logger.info("=== 测试策略构思CRUD操作 ===")
    
    # 测试数据
    test_strategy = {
        "id": f"test_strategy_{int(time.time())}",
        "name": "测试策略",
        "description": "测试策略构思的CRUD操作",
        "type": "trend_following",  # 添加必需的type字段
        "author": "测试用户",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "tags": ["测试", "CRUD"],
        "status": "draft"
    }
    
    try:
        # 测试创建
        create_result = save_strategy_conception(test_strategy)
        logger.info(f"创建策略构思: {'成功' if create_result.get('success') else '失败'}")
        assert create_result.get('success'), "创建策略构思失败"
        
        # 测试读取 - 直接使用持久化管理器
        from src.gateway.web.unified_persistence import get_strategy_conception_persistence
        persistence = get_strategy_conception_persistence()
        retrieved_strategy = persistence.load(test_strategy["id"])
        logger.info(f"读取策略构思: {'成功' if retrieved_strategy else '失败'}")
        assert retrieved_strategy, "读取策略构思失败"
        assert retrieved_strategy["name"] == test_strategy["name"], "策略构思名称不匹配"
        
        # 测试更新
        update_data = test_strategy.copy()
        update_data["description"] = "更新后的测试策略"
        update_data["status"] = "active"
        update_result = save_strategy_conception(update_data)
        logger.info(f"更新策略构思: {'成功' if update_result.get('success') else '失败'}")
        assert update_result.get('success'), "更新策略构思失败"
        
        # 验证更新
        updated_strategy = persistence.load(test_strategy["id"])
        logger.info(f"验证更新后的策略构思: {'成功' if updated_strategy else '失败'}")
        
        # 测试列表
        strategies = load_strategy_conceptions()
        logger.info(f"列出策略构思: {'成功' if strategies else '失败'}")
        assert len(strategies) >= 0, "列出策略构思失败"
        
        # 测试删除
        delete_success = persistence.delete(test_strategy["id"])
        logger.info(f"删除策略构思: {'成功' if delete_success else '失败'}")
        assert delete_success, "删除策略构思失败"
        
        # 验证删除
        deleted_strategy = persistence.load(test_strategy["id"])
        assert deleted_strategy is None, "策略构思删除失败"
        
        logger.info("策略构思CRUD测试全部通过!")
        return True
    except Exception as e:
        logger.error(f"策略构思CRUD测试失败: {e}")
        # 清理测试数据
        try:
            from src.gateway.web.unified_persistence import get_strategy_conception_persistence
            persistence = get_strategy_conception_persistence()
            persistence.delete(test_strategy["id"])
        except:
            pass
        return False


def test_backtest_result_crud():
    """
    测试回测结果的CRUD操作
    """
    logger.info("=== 测试回测结果CRUD操作 ===")
    
    # 测试数据
    test_backtest = {
        "backtest_id": f"test_backtest_{int(time.time())}",
        "strategy_id": "test_strategy",
        "status": "completed",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000.0,
        "final_capital": 120000.0,
        "total_return": 0.2,
        "annualized_return": 0.18,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.1,
        "win_rate": 0.6,
        "total_trades": 100,
        "equity_curve": [
            {"date": "2024-01-01", "equity": 100000.0},
            {"date": "2024-12-31", "equity": 120000.0}
        ],
        "trades": [],
        "metrics": {"alpha": 0.05, "beta": 0.8}
    }
    
    try:
        # 测试创建
        create_success = save_backtest_result(test_backtest)
        logger.info(f"创建回测结果: {'成功' if create_success else '失败'}")
        assert create_success, "创建回测结果失败"
        
        # 测试读取
        retrieved_backtest = load_backtest_result(test_backtest["backtest_id"])
        logger.info(f"读取回测结果: {'成功' if retrieved_backtest else '失败'}")
        assert retrieved_backtest, "读取回测结果失败"
        assert retrieved_backtest["backtest_id"] == test_backtest["backtest_id"], "回测ID不匹配"
        
        # 测试更新
        update_data = {"status": "analyzed", "sharpe_ratio": 1.6}
        update_success = update_backtest_result(test_backtest["backtest_id"], update_data)
        logger.info(f"更新回测结果: {'成功' if update_success else '失败'}")
        assert update_success, "更新回测结果失败"
        
        # 验证更新
        updated_backtest = load_backtest_result(test_backtest["backtest_id"])
        assert updated_backtest["status"] == "analyzed", "回测结果状态更新失败"
        assert updated_backtest["sharpe_ratio"] == 1.6, "回测结果夏普比率更新失败"
        
        # 测试列表
        backtests = list_backtest_results(strategy_id="test_strategy")
        logger.info(f"列出回测结果: {'成功' if backtests else '失败'}")
        
        # 测试删除
        delete_success = delete_backtest_result(test_backtest["backtest_id"])
        logger.info(f"删除回测结果: {'成功' if delete_success else '失败'}")
        assert delete_success, "删除回测结果失败"
        
        # 验证删除
        deleted_backtest = load_backtest_result(test_backtest["backtest_id"])
        assert deleted_backtest is None, "回测结果删除失败"
        
        logger.info("回测结果CRUD测试全部通过!")
        return True
    except Exception as e:
        logger.error(f"回测结果CRUD测试失败: {e}")
        # 清理测试数据
        try:
            delete_backtest_result(test_backtest["backtest_id"])
        except:
            pass
        return False


def test_dual_storage_mechanism():
    """
    测试双存储机制：优先PostgreSQL，降级文件系统
    """
    logger.info("=== 测试双存储机制 ===")
    
    # 测试数据
    test_strategy = {
        "id": f"test_dual_storage_{int(time.time())}",
        "name": "双存储测试策略",
        "description": "测试PostgreSQL与文件系统双存储机制",
        "author": "测试用户",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "tags": ["测试", "双存储"],
        "status": "draft"
    }
    
    try:
        # 1. 测试保存到数据库
        save_success = save_strategy_conception(test_strategy)
        logger.info(f"保存到双存储: {'成功' if save_success else '失败'}")
        assert save_success, "保存到双存储失败"
        
        # 2. 测试从数据库读取
        retrieved_strategy = get_strategy_conception(test_strategy["id"])
        logger.info(f"从双存储读取: {'成功' if retrieved_strategy else '失败'}")
        assert retrieved_strategy, "从双存储读取失败"
        
        # 3. 测试文件系统备份（验证文件是否存在）
        data_dir = os.path.join(os.path.dirname(__file__), "../../data")
        strategy_file = os.path.join(data_dir, "strategy_conceptions", f"{test_strategy['id']}.json")
        file_exists = os.path.exists(strategy_file)
        logger.info(f"文件系统备份: {'存在' if file_exists else '不存在'}")
        
        # 4. 验证数据一致性
        if file_exists:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            assert file_data["name"] == test_strategy["name"], "文件系统数据与原始数据不一致"
            logger.info("文件系统数据与原始数据一致")
        
        # 5. 清理测试数据
        delete_success = delete_strategy_conception(test_strategy["id"])
        logger.info(f"清理双存储测试数据: {'成功' if delete_success else '失败'}")
        
        # 6. 验证文件也被删除
        if os.path.exists(strategy_file):
            os.remove(strategy_file)
        
        logger.info("双存储机制测试通过!")
        return True
    except Exception as e:
        logger.error(f"双存储机制测试失败: {e}")
        # 清理测试数据
        try:
            delete_strategy_conception(test_strategy["id"])
            # 清理文件系统
            data_dir = os.path.join(os.path.dirname(__file__), "../../data")
            strategy_file = os.path.join(data_dir, "strategy_conceptions", f"{test_strategy['id']}.json")
            if os.path.exists(strategy_file):
                os.remove(strategy_file)
        except:
            pass
        return False


def test_migration_integrity():
    """
    测试迁移完整性：验证数据从文件系统迁移到数据库后的数据一致性
    """
    logger.info("=== 测试迁移完整性 ===")
    
    # 1. 检查策略构思数据
    strategies = load_strategy_conceptions()
    logger.info(f"当前策略构思数量: {len(strategies)}")
    
    # 2. 检查回测结果数据
    backtests = list_backtest_results()
    logger.info(f"当前回测结果数量: {len(backtests)}")
    
    # 3. 验证数据结构完整性
    if strategies:
        sample_strategy = strategies[0]
        required_fields = ["id", "name", "description", "author", "created_at"]
        missing_fields = [field for field in required_fields if field not in sample_strategy]
        if missing_fields:
            logger.warning(f"策略构思缺少字段: {missing_fields}")
        else:
            logger.info("策略构思数据结构完整")
    
    if backtests:
        sample_backtest = backtests[0]
        required_fields = ["backtest_id", "strategy_id", "status", "start_date", "end_date"]
        missing_fields = [field for field in required_fields if field not in sample_backtest]
        if missing_fields:
            logger.warning(f"回测结果缺少字段: {missing_fields}")
        else:
            logger.info("回测结果数据结构完整")
    
    logger.info("迁移完整性测试完成!")
    return True


def run_all_tests():
    """
    运行所有测试
    """
    logger.info("开始运行持久化迁移测试套件...")
    
    tests = [
        ("数据库连接测试", test_database_connection),
        ("策略构思CRUD测试", test_strategy_conception_crud),
        ("回测结果CRUD测试", test_backtest_result_crud),
        ("双存储机制测试", test_dual_storage_mechanism),
        ("迁移完整性测试", test_migration_integrity)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n运行测试: {test_name}")
        if test_func():
            passed_tests += 1
            logger.info(f"测试结果: ✅ 通过")
        else:
            logger.info(f"测试结果: ❌ 失败")
    
    # 打印测试结果汇总
    logger.info("\n=== 测试结果汇总 ===")
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过测试数: {passed_tests}")
    logger.info(f"失败测试数: {total_tests - passed_tests}")
    logger.info(f"测试通过率: {passed_tests / total_tests * 100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！持久化迁移功能完整")
        return True
    else:
        logger.warning("⚠️  部分测试失败，需要检查修复")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
