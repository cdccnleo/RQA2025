"""
测试回测结果PostgreSQL持久化功能
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_postgresql_connection():
    """测试PostgreSQL连接"""
    print("=" * 60)
    print("测试PostgreSQL连接")
    print("=" * 60)
    
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection, get_db_config
        
        config = get_db_config()
        print(f"数据库配置:")
        print(f"  主机: {config.get('host')}")
        print(f"  端口: {config.get('port')}")
        print(f"  数据库: {config.get('database')}")
        print(f"  用户: {config.get('user')}")
        print(f"  密码: {'*' * len(config.get('password', '')) if config.get('password') else '(未设置)'}")
        
        conn = get_db_connection()
        if conn:
            print("✅ PostgreSQL连接成功")
            
            # 测试查询
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ PostgreSQL版本: {version}")
            
            cursor.close()
            return_db_connection(conn)
            return True
        else:
            print("❌ PostgreSQL连接失败")
            return False
    except Exception as e:
        print(f"❌ PostgreSQL连接测试失败: {e}")
        return False


def test_save_backtest_result():
    """测试保存回测结果到PostgreSQL"""
    print("\n" + "=" * 60)
    print("测试保存回测结果到PostgreSQL")
    print("=" * 60)
    
    try:
        from src.gateway.web.backtest_persistence import save_backtest_result
        
        # 创建测试回测结果
        test_backtest = {
            "backtest_id": f"test_backtest_{int(datetime.now().timestamp())}",
            "strategy_id": "test_strategy_1",
            "status": "completed",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000.0,
            "final_capital": 120000.0,
            "total_return": 0.20,
            "annualized_return": 0.20,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "win_rate": 0.60,
            "total_trades": 100,
            "equity_curve": [100000, 105000, 110000, 115000, 120000],
            "trades": [
                {"symbol": "000001", "action": "buy", "price": 10.0, "quantity": 1000},
                {"symbol": "000001", "action": "sell", "price": 12.0, "quantity": 1000}
            ],
            "metrics": {
                "volatility": 0.15,
                "beta": 1.0
            },
            "created_at": datetime.now().isoformat()
        }
        
        result = save_backtest_result(test_backtest)
        if result:
            print(f"✅ 回测结果已保存: {test_backtest['backtest_id']}")
            return test_backtest['backtest_id']
        else:
            print("❌ 保存回测结果失败")
            return None
    except Exception as e:
        print(f"❌ 保存回测结果测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_backtest_result(backtest_id):
    """测试从PostgreSQL加载回测结果"""
    print("\n" + "=" * 60)
    print("测试从PostgreSQL加载回测结果")
    print("=" * 60)
    
    if not backtest_id:
        print("⚠️ 跳过测试（没有可用的backtest_id）")
        return False
    
    try:
        from src.gateway.web.backtest_persistence import load_backtest_result
        
        result = load_backtest_result(backtest_id)
        if result:
            print(f"✅ 成功加载回测结果: {backtest_id}")
            print(f"   策略ID: {result.get('strategy_id')}")
            print(f"   状态: {result.get('status')}")
            print(f"   总收益: {result.get('total_return')}")
            print(f"   夏普比率: {result.get('sharpe_ratio')}")
            return True
        else:
            print(f"❌ 加载回测结果失败: {backtest_id}")
            return False
    except Exception as e:
        print(f"❌ 加载回测结果测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_backtest_results():
    """测试列出回测结果"""
    print("\n" + "=" * 60)
    print("测试列出回测结果")
    print("=" * 60)
    
    try:
        from src.gateway.web.backtest_persistence import list_backtest_results
        
        results = list_backtest_results(limit=10)
        print(f"✅ 成功列出 {len(results)} 个回测结果")
        
        if results:
            print("\n前5个回测结果:")
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result.get('backtest_id')} - {result.get('strategy_id')} - {result.get('status')}")
        
        return True
    except Exception as e:
        print(f"❌ 列出回测结果测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postgresql_table_exists():
    """测试PostgreSQL表是否存在"""
    print("\n" + "=" * 60)
    print("测试PostgreSQL表结构")
    print("=" * 60)
    
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到PostgreSQL")
            return False
        
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'backtest_results'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("✅ backtest_results 表存在")
            
            # 获取表结构
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'backtest_results'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            print("\n表结构:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
            # 获取记录数
            cursor.execute("SELECT COUNT(*) FROM backtest_results;")
            count = cursor.fetchone()[0]
            print(f"\n记录数: {count}")
        else:
            print("⚠️ backtest_results 表不存在（将在首次保存时自动创建）")
        
        cursor.close()
        return_db_connection(conn)
        return True
    except Exception as e:
        print(f"❌ 检查表结构失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("回测结果PostgreSQL持久化功能测试")
    print("=" * 60)
    
    results = {
        "连接测试": False,
        "表结构测试": False,
        "保存测试": False,
        "加载测试": False,
        "列表测试": False
    }
    
    # 1. 测试连接
    results["连接测试"] = test_postgresql_connection()
    
    # 2. 测试表结构
    if results["连接测试"]:
        results["表结构测试"] = test_postgresql_table_exists()
    
    # 3. 测试保存
    backtest_id = None
    if results["连接测试"]:
        backtest_id = test_save_backtest_result()
        results["保存测试"] = backtest_id is not None
    
    # 4. 测试加载
    if backtest_id:
        results["加载测试"] = test_load_backtest_result(backtest_id)
    
    # 5. 测试列表
    if results["连接测试"]:
        results["列表测试"] = test_list_backtest_results()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ 所有测试通过！PostgreSQL持久化功能正常。")
    else:
        print("\n⚠️ 部分测试失败。")
        if not results["连接测试"]:
            print("   - PostgreSQL连接失败，请检查数据库配置和连接")
        if results["连接测试"] and not results["保存测试"]:
            print("   - 保存功能可能有问题，但文件系统存储仍然可用")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

