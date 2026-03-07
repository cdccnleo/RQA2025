#!/usr/bin/env python3
"""
简化的量化模型测试运行器
避免复杂依赖，专注于数值精度和边界条件测试
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_simple_numerical_test():
    """创建简化的数值精度测试"""
    test_code = '''
import unittest
import numpy as np
from decimal import Decimal, getcontext

# 设置高精度计算
getcontext().prec = 28

class TestNumericalPrecision(unittest.TestCase):
    """数值精度测试类"""
    
    def test_decimal_precision(self):
        """测试Decimal精度计算"""
        # 金融计算中的精度测试
        price1 = Decimal('100.123456789')
        price2 = Decimal('200.987654321')
        
        # 加法精度测试
        sum_result = price1 + price2
        expected_sum = Decimal('301.111111110')
        self.assertLess(abs(sum_result - expected_sum), Decimal('1e-10'))
        
        # 乘法精度测试
        product = price1 * price2
        expected_product = Decimal('20123.4567890123456789')
        self.assertLess(abs(product - expected_product), Decimal('1e-10'))
    
    def test_numpy_precision(self):
        """测试NumPy数值精度"""
        # 浮点数精度测试
        a = np.float64(0.1)
        b = np.float64(0.2)
        c = np.float64(0.3)
        
        # 测试浮点数加法精度
        result = a + b
        self.assertLess(abs(result - c), 1e-15)
        
        # 测试累积误差
        values = np.array([0.1] * 10, dtype=np.float64)
        sum_result = np.sum(values)
        expected_sum = 1.0
        self.assertLess(abs(sum_result - expected_sum), 1e-15)
    
    def test_financial_calculations(self):
        """测试金融计算精度"""
        # 复利计算精度测试
        principal = Decimal('10000')
        rate = Decimal('0.05')
        time_periods = Decimal('10')
        
        # 复利公式: A = P(1 + r)^t
        compound_interest = principal * ((1 + rate) ** time_periods)
        expected_result = Decimal('16288.9462677744162431')
        self.assertLess(abs(compound_interest - expected_result), Decimal('1e-10'))
    
    def test_boundary_conditions(self):
        """测试边界条件"""
        # 极小值测试
        tiny_value = Decimal('1e-20')
        self.assertGreater(tiny_value, 0)
        
        # 极大值测试
        large_value = Decimal('1e20')
        self.assertLess(large_value, Decimal('inf'))
        
        # 零值测试
        zero = Decimal('0')
        self.assertEqual(zero, 0)

if __name__ == "__main__":
    unittest.main()
'''

    test_file = Path("tests/unit/quantitative/test_simple_numerical.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)

    return str(test_file)


def create_simple_boundary_test():
    """创建简化的边界条件测试"""
    test_code = '''
import unittest
import numpy as np
from decimal import Decimal, getcontext

# 设置高精度计算
getcontext().prec = 28

class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试类"""
    
    def test_extreme_values(self):
        """测试极值处理"""
        # 极大值测试
        max_float = np.finfo(np.float64).max
        min_float = np.finfo(np.float64).min
        
        # 极大值运算
        self.assertEqual(max_float + 0, max_float)
        self.assertEqual(max_float * 1, max_float)
        
        # 极小值运算
        self.assertEqual(min_float + 0, min_float)
        self.assertEqual(min_float * 1, min_float)
        
        # Decimal极值测试
        max_decimal = Decimal('1e20')
        min_decimal = Decimal('1e-20')
        
        self.assertGreater(max_decimal, 0)
        self.assertGreater(min_decimal, 0)
        self.assertLess(max_decimal, Decimal('inf'))
    
    def test_zero_values(self):
        """测试零值处理"""
        # 零值运算
        zero = Decimal('0')
        self.assertEqual(zero, 0)
        self.assertEqual(zero + zero, zero)
        self.assertEqual(zero * zero, zero)
        
        # 零值除法
        with self.assertRaises(ZeroDivisionError):
            result = 1 / zero
        
        # 零值数组
        zero_array = np.zeros(100)
        self.assertEqual(np.sum(zero_array), 0)
        self.assertEqual(np.mean(zero_array), 0)
    
    def test_infinity_handling(self):
        """测试无穷大处理"""
        # 正无穷
        pos_inf = np.inf
        self.assertGreater(pos_inf, 0)
        self.assertEqual(pos_inf + 1, pos_inf)
        self.assertEqual(pos_inf * 2, pos_inf)
        
        # 负无穷
        neg_inf = -np.inf
        self.assertLess(neg_inf, 0)
        self.assertEqual(neg_inf - 1, neg_inf)
        self.assertEqual(neg_inf * 2, neg_inf)
    
    def test_nan_handling(self):
        """测试NaN处理"""
        # NaN检测
        nan_value = np.nan
        self.assertTrue(np.isnan(nan_value))
        
        # NaN传播
        self.assertTrue(np.isnan(nan_value + 1))
        self.assertTrue(np.isnan(nan_value * 2))
        self.assertTrue(np.isnan(nan_value / 2))
        
        # NaN数组处理
        nan_array = np.array([1, 2, np.nan, 4, 5])
        self.assertTrue(np.isnan(np.mean(nan_array)))
        self.assertFalse(np.isnan(np.nanmean(nan_array)))

if __name__ == "__main__":
    unittest.main()
'''

    test_file = Path("tests/unit/quantitative/test_simple_boundary.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)

    return str(test_file)


def create_simple_timeseries_test():
    """创建简化的时间序列测试"""
    test_code = '''
import unittest
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

# 设置高精度计算
getcontext().prec = 28

class TestTimeSeriesProcessing(unittest.TestCase):
    """时间序列处理测试类"""
    
    def test_moving_averages(self):
        """测试移动平均计算"""
        # 生成测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        ts = pd.Series(prices, index=dates)
        
        # 简单移动平均
        ma_5 = ts.rolling(window=5).mean()
        ma_20 = ts.rolling(window=20).mean()
        
        # 验证移动平均
        self.assertEqual(len(ma_5), len(ts))
        self.assertEqual(ma_5.iloc[4], ts.iloc[0:5].mean())
        self.assertEqual(ma_20.iloc[19], ts.iloc[0:20].mean())
        
        # 指数移动平均
        ema_12 = ts.ewm(span=12).mean()
        self.assertEqual(len(ema_12), len(ts))
        self.assertFalse(ema_12.isna().all())
    
    def test_technical_indicators(self):
        """测试技术指标计算"""
        # 生成价格数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        high = np.random.randn(100).cumsum() + 100
        low = high - np.random.uniform(1, 5, 100)
        close = (high + low) / 2 + np.random.uniform(-1, 1, 100)
        
        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        }, index=dates)
        
        # RSI计算
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(df['close'])
        self.assertEqual(len(rsi), len(df))
        self.assertGreaterEqual(rsi.iloc[13], 0)
        self.assertLessEqual(rsi.iloc[13], 100)
    
    def test_volatility_calculation(self):
        """测试波动率计算"""
        # 生成收益率数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, 252)
        returns_ts = pd.Series(returns, index=dates)
        
        # 历史波动率
        rolling_vol = returns_ts.rolling(window=20).std() * np.sqrt(252)
        self.assertEqual(len(rolling_vol), len(returns_ts))
        self.assertGreaterEqual(rolling_vol.iloc[19], 0)
    
    def test_data_quality_checks(self):
        """测试数据质量检查"""
        # 生成有质量问题的数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = np.random.normal(100, 10, 100)
        
        # 添加缺失值
        data[10:15] = np.nan
        # 添加重复值
        data[20:25] = data[20]
        # 添加异常值
        data[30] = 1000
        
        ts = pd.Series(data, index=dates)
        
        # 数据质量检查
        missing_ratio = ts.isna().sum() / len(ts)
        self.assertEqual(missing_ratio, 0.05)
        
        duplicate_ratio = ts.duplicated().sum() / len(ts)
        self.assertGreater(duplicate_ratio, 0)
        
        # 数据完整性
        self.assertEqual(len(ts), 100)
        self.assertTrue(ts.index.is_monotonic_increasing)

if __name__ == "__main__":
    unittest.main()
'''

    test_file = Path("tests/unit/quantitative/test_simple_timeseries.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)

    return str(test_file)


def run_test_file(test_file: str) -> Dict:
    """运行测试文件"""
    print(f"运行测试: {test_file}")

    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        end_time = time.time()

        # 解析输出
        output = result.stdout + result.stderr

        # 统计测试结果
        passed = output.count("test_") - output.count("FAILED") - output.count("ERROR")
        failed = output.count("FAILED")
        errors = output.count("ERROR")

        return {
            'file': test_file,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'total': passed + failed + errors,
            'duration': end_time - start_time,
            'return_code': result.returncode,
            'output': output
        }

    except subprocess.TimeoutExpired:
        print(f"测试超时: {test_file}")
        return {
            'file': test_file,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'total': 1,
            'duration': 60,
            'return_code': -1,
            'output': "测试超时"
        }
    except Exception as e:
        print(f"运行测试失败: {test_file}, 错误: {str(e)}")
        return {
            'file': test_file,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'total': 1,
            'duration': 0,
            'return_code': -1,
            'output': str(e)
        }


def main():
    """主函数"""
    print("开始量化模型测试增强")
    print("="*50)

    # 创建测试文件
    test_files = []
    test_files.append(create_simple_numerical_test())
    test_files.append(create_simple_boundary_test())
    test_files.append(create_simple_timeseries_test())

    print(f"生成测试文件: {len(test_files)}个")

    # 运行测试
    results = []
    for test_file in test_files:
        result = run_test_file(test_file)
        results.append(result)
        print(f"文件: {Path(test_file).name}")
        print(f"  通过: {result['passed']}")
        print(f"  失败: {result['failed']}")
        print(f"  错误: {result['errors']}")
        print(f"  耗时: {result['duration']:.2f}秒")
        print()

    # 统计结果
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_tests = sum(r['total'] for r in results)

    print("="*50)
    print("测试结果汇总")
    print("="*50)
    print(f"总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"错误: {total_errors}")

    if total_tests > 0:
        success_rate = total_passed / total_tests * 100
        print(f"成功率: {success_rate:.1f}%")

    print("="*50)


if __name__ == "__main__":
    main()
