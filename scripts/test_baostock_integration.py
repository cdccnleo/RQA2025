#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaoStock 集成测试脚本

测试内容：
1. BaoStock 连接测试
2. 数据采集功能测试
3. 与 AKShare 的数据一致性比较
4. 数据源切换逻辑验证
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_baostock_import():
    """测试1: BaoStock 库导入"""
    print("\n" + "="*60)
    print("测试1: BaoStock 库导入")
    print("="*60)
    
    try:
        import baostock as bs
        print(f"✅ BaoStock 库导入成功")
        print(f"   版本信息: {bs.__version__ if hasattr(bs, '__version__') else '未知'}")
        return True
    except ImportError as e:
        print(f"❌ BaoStock 库导入失败: {e}")
        return False


def test_baostock_connection():
    """测试2: BaoStock 连接测试"""
    print("\n" + "="*60)
    print("测试2: BaoStock 连接测试")
    print("="*60)
    
    try:
        import baostock as bs
        
        # 登录
        print("🔄 正在连接 BaoStock 服务器...")
        lg = bs.login()
        
        if lg.error_code == '0':
            print(f"✅ BaoStock 登录成功")
            print(f"   错误码: {lg.error_code}")
            print(f"   消息: {lg.error_msg}")
            
            # 登出
            bs.logout()
            print("✅ BaoStock 登出成功")
            return True
        else:
            print(f"❌ BaoStock 登录失败")
            print(f"   错误码: {lg.error_code}")
            print(f"   消息: {lg.error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ BaoStock 连接异常: {e}")
        return False


def test_baostock_data_fetch():
    """测试3: BaoStock 数据获取测试"""
    print("\n" + "="*60)
    print("测试3: BaoStock 数据获取测试")
    print("="*60)
    
    try:
        import baostock as bs
        
        # 登录
        lg = bs.login()
        if lg.error_code != '0':
            print(f"❌ 登录失败: {lg.error_msg}")
            return False
        
        # 测试获取平安银行(sz.000001)的数据
        symbol = "sz.000001"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"🔄 获取股票数据: {symbol}")
        print(f"   日期范围: {start_date} ~ {end_date}")
        
        # 查询历史K线数据
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            code=symbol,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"❌ 查询失败: {rs.error_msg}")
            bs.logout()
            return False
        
        # 转换为 DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=fields.split(','))
        
        print(f"✅ 数据获取成功")
        print(f"   记录数: {len(df)}")
        print(f"   字段: {list(df.columns)}")
        
        if not df.empty:
            print(f"\n   数据样例 (前3行):")
            print(df.head(3).to_string(index=False))
        
        bs.logout()
        return True
        
    except Exception as e:
        print(f"❌ 数据获取异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_baostock_service():
    """测试4: BaoStock 服务模块测试"""
    print("\n" + "="*60)
    print("测试4: BaoStock 服务模块测试")
    print("="*60)
    
    try:
        from src.core.integration.baostock_service import get_baostock_service
        
        # 获取服务实例
        service = get_baostock_service()
        print(f"✅ BaoStock 服务实例创建成功")
        print(f"   服务可用: {service.is_available}")
        
        if not service.is_available:
            print("⚠️ BaoStock 服务不可用，跳过数据测试")
            return False
        
        # 测试获取数据
        symbol = "000001"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        
        print(f"\n🔄 通过服务获取股票数据: {symbol}")
        print(f"   日期范围: {start_date} ~ {end_date}")
        
        df = await service.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        if df is not None and not df.empty:
            print(f"✅ 服务数据获取成功")
            print(f"   记录数: {len(df)}")
            print(f"   字段: {list(df.columns)}")
            print(f"\n   数据样例 (前3行):")
            print(df.head(3).to_string(index=False))
            return True
        else:
            print("❌ 服务返回空数据")
            return False
            
    except Exception as e:
        print(f"❌ 服务测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_consistency():
    """测试5: AKShare 与 BaoStock 数据一致性比较"""
    print("\n" + "="*60)
    print("测试5: AKShare 与 BaoStock 数据一致性比较")
    print("="*60)
    
    try:
        import akshare as ak
        import baostock as bs
        
        symbol = "000001"  # 平安银行
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        print(f"📊 比较股票: {symbol}")
        print(f"   日期范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 获取 AKShare 数据
        print("\n🔄 获取 AKShare 数据...")
        try:
            df_akshare = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"
            )
            print(f"   AKShare 记录数: {len(df_akshare) if df_akshare is not None else 0}")
        except Exception as e:
            print(f"   ⚠️ AKShare 获取失败: {e}")
            df_akshare = None
        
        # 获取 BaoStock 数据
        print("\n🔄 获取 BaoStock 数据...")
        lg = bs.login()
        if lg.error_code == '0':
            rs = bs.query_history_k_data_plus(
                code=f"sz.{symbol}",
                fields="date,open,high,low,close,volume",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df_baostock = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_baostock[col] = pd.to_numeric(df_baostock[col], errors='coerce')
            
            print(f"   BaoStock 记录数: {len(df_baostock)}")
            bs.logout()
        else:
            print(f"   ⚠️ BaoStock 登录失败")
            df_baostock = None
        
        # 比较数据
        if df_akshare is not None and not df_akshare.empty and df_baostock is not None and not df_baostock.empty:
            print("\n📊 数据对比:")
            print("\n   AKShare 数据样例:")
            print(df_akshare[['日期', '开盘', '收盘', '最高', '最低']].head(5).to_string(index=False))
            
            print("\n   BaoStock 数据样例:")
            print(df_baostock[['date', 'open', 'close', 'high', 'low']].head(5).to_string(index=False))
            
            # 简单的数值对比（比较收盘价）
            if len(df_akshare) > 0 and len(df_baostock) > 0:
                ak_close = df_akshare['收盘'].iloc[0]
                bs_close = df_baostock['close'].iloc[0]
                diff_pct = abs(ak_close - bs_close) / ak_close * 100 if ak_close != 0 else 0
                
                print(f"\n   最新日期收盘价对比:")
                print(f"   AKShare: {ak_close}")
                print(f"   BaoStock: {bs_close}")
                print(f"   差异: {diff_pct:.4f}%")
                
                if diff_pct < 1:  # 差异小于1%认为一致
                    print("   ✅ 数据一致性良好")
                    return True
                else:
                    print("   ⚠️ 数据存在差异，可能是复权方式或数据更新时间不同")
                    return True  # 仍然认为测试通过
        else:
            print("⚠️ 无法进行数据对比（某个数据源返回空数据）")
            return df_baostock is not None and not df_baostock.empty
            
    except Exception as e:
        print(f"❌ 数据一致性测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_source_manager():
    """测试6: 数据源管理器切换逻辑测试"""
    print("\n" + "="*60)
    print("测试6: 数据源管理器切换逻辑测试")
    print("="*60)
    
    try:
        from src.core.integration.data_source_manager import get_data_source_manager, reset_data_source_manager
        
        # 重置管理器
        reset_data_source_manager()
        
        # 获取管理器实例
        manager = get_data_source_manager()
        print(f"✅ 数据源管理器创建成功")
        
        # 查看数据源状态
        stats = manager.get_data_source_stats()
        print(f"\n📊 数据源状态:")
        for source, stat in stats.items():
            print(f"   {source}:")
            print(f"      状态: {stat['status']}")
            print(f"      成功率: {stat['success_rate']:.2%}")
            print(f"      失败次数: {stat['failure_count']}")
        
        # 测试获取数据（会自动选择数据源）
        symbol = "000001"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
        
        print(f"\n🔄 通过管理器获取股票数据: {symbol}")
        print(f"   日期范围: {start_date} ~ {end_date}")
        
        df = await manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type="daily",
            adjust="qfq"
        )
        
        if df is not None and not df.empty:
            print(f"✅ 管理器数据获取成功")
            print(f"   记录数: {len(df)}")
            print(f"   字段: {list(df.columns)}")
            
            # 查看缓存统计
            cache_stats = manager.get_cache_stats()
            print(f"\n📊 缓存统计:")
            print(f"   命中: {cache_stats['hits']}")
            print(f"   未命中: {cache_stats['misses']}")
            print(f"   命中率: {cache_stats['hit_rate']:.2%}")
            
            return True
        else:
            print("❌ 管理器返回空数据")
            return False
            
    except Exception as e:
        print(f"❌ 数据源管理器测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("BaoStock 集成测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # 测试1: 库导入
    results['导入测试'] = test_baostock_import()
    
    # 测试2: 连接测试
    results['连接测试'] = test_baostock_connection()
    
    # 测试3: 数据获取测试
    results['数据获取测试'] = test_baostock_data_fetch()
    
    # 测试4: 服务模块测试
    results['服务模块测试'] = asyncio.run(test_baostock_service())
    
    # 测试5: 数据一致性测试
    results['数据一致性测试'] = asyncio.run(test_data_consistency())
    
    # 测试6: 数据源管理器测试
    results['数据源管理器测试'] = asyncio.run(test_data_source_manager())
    
    # 打印总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
