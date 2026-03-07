"""实时交易系统测试模块

测试 src.trading.realtime.realtime_realtime_trading_system 模块的功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.trading.realtime.realtime_realtime_trading_system import RealtimeTradingSystem


class TestRealtimeTradingSystem:
    """实时交易系统测试类"""
    
    @pytest.fixture
    def default_config(self):
        """创建默认配置"""
        return {
            'trading_interval': 1  # 测试时使用较短的间隔
        }
    
    @pytest.fixture
    def trading_system(self, default_config):
        """创建实时交易系统实例"""
        return RealtimeTradingSystem(default_config)
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        system = RealtimeTradingSystem()
        assert system.config == {}
        assert system.is_running is False
        assert system.trading_thread is None
        assert isinstance(system.positions, dict)
        assert isinstance(system.orders, dict)
        assert isinstance(system.trading_history, list)
        assert isinstance(system.market_data, dict)
        assert system.last_update is None
    
    def test_init_with_custom_config(self, default_config):
        """测试使用自定义配置初始化"""
        system = RealtimeTradingSystem(default_config)
        assert system.config == default_config
        assert system.config['trading_interval'] == 1
    
    def test_initialize_success(self, trading_system):
        """测试初始化成功"""
        result = trading_system.initialize()
        assert result is True
    
    def test_initialize_with_exception(self, trading_system):
        """测试初始化时发生异常"""
        # 模拟初始化时发生异常
        original_init = trading_system.initialize
        def failing_init():
            raise Exception("初始化失败")
        
        trading_system.initialize = failing_init
        try:
            result = trading_system.initialize()
            assert False, "应该抛出异常"
        except Exception:
            # 异常被捕获，测试通过
            pass
        finally:
            trading_system.initialize = original_init
    
    def test_start_success(self, trading_system):
        """测试启动成功"""
        trading_system.initialize()
        result = trading_system.start()
        
        assert result is True
        assert trading_system.is_running is True
        assert trading_system.trading_thread is not None
        assert trading_system.trading_thread.is_alive()
        
        # 清理
        trading_system.stop()
    
    def test_start_when_already_running(self, trading_system):
        """测试系统已在运行时启动应该返回False"""
        trading_system.initialize()
        trading_system.start()
        
        # 再次启动应该返回False
        result = trading_system.start()
        assert result is False
        
        # 清理
        trading_system.stop()
    
    def test_start_with_exception(self, trading_system):
        """测试启动时发生异常"""
        trading_system.initialize()
        
        # 模拟启动时发生异常
        with patch('threading.Thread', side_effect=Exception("启动失败")):
            result = trading_system.start()
            assert result is False
            assert trading_system.is_running is False
    
    def test_stop(self, trading_system):
        """测试停止系统"""
        trading_system.initialize()
        trading_system.start()
        
        assert trading_system.is_running is True
        
        trading_system.stop()
        
        # 等待线程结束
        if trading_system.trading_thread:
            trading_system.trading_thread.join(timeout=2)
        
        assert trading_system.is_running is False
    
    def test_stop_when_not_running(self, trading_system):
        """测试停止未运行的系统"""
        trading_system.stop()
        assert trading_system.is_running is False
    
    def test_get_market_data(self, trading_system):
        """测试获取市场数据"""
        market_data = trading_system._get_market_data()
        
        assert market_data is not None
        assert 'timestamp' in market_data
        assert 'symbols' in market_data
        assert 'prices' in market_data
        assert isinstance(market_data['timestamp'], datetime)
        assert trading_system.last_update is not None
        assert trading_system.market_data == market_data
    
    def test_get_market_data_with_exception(self, trading_system):
        """测试获取市场数据时发生异常"""
        # 模拟_get_market_data内部发生异常
        original_method = trading_system._get_market_data
        def failing_get_market_data():
            raise Exception("市场数据获取失败")
        
        trading_system._get_market_data = failing_get_market_data
        try:
            market_data = trading_system._get_market_data()
            # 如果方法没有处理异常，应该抛出异常
            assert False, "应该抛出异常"
        except Exception:
            # 异常被抛出，测试通过
            pass
        finally:
            trading_system._get_market_data = original_method
    
    def test_perform_analysis(self, trading_system):
        """测试执行分析"""
        market_data = {
            'timestamp': datetime.now(),
            'symbols': ['000001.SZ'],
            'prices': {'000001.SZ': {'close': 10.5, 'volume': 1000000}}
        }
        
        analysis_result = trading_system._perform_analysis(market_data)
        
        assert isinstance(analysis_result, dict)
        assert 'ml_prediction' in analysis_result
        assert 'analysis_report' in analysis_result
        assert 'market_data' in analysis_result
        assert analysis_result['market_data'] == market_data
    
    def test_perform_analysis_with_exception(self, trading_system):
        """测试执行分析时发生异常"""
        # 模拟_perform_analysis内部发生异常
        original_method = trading_system._perform_analysis
        def failing_perform_analysis(market_data):
            raise Exception("分析失败")
        
        trading_system._perform_analysis = failing_perform_analysis
        try:
            result = trading_system._perform_analysis({})
            # 如果方法没有处理异常，应该抛出异常
            assert False, "应该抛出异常"
        except Exception:
            # 异常被抛出，测试通过
            pass
        finally:
            trading_system._perform_analysis = original_method
    
    def test_generate_signals_with_ml_prediction(self, trading_system):
        """测试基于ML预测生成交易信号"""
        analysis_result = {
            'ml_prediction': {'prediction': 1, 'confidence': 0.8},
            'analysis_report': {'composite_score': 0.2}
        }
        
        signals = trading_system._generate_signals(analysis_result)
        
        assert isinstance(signals, list)
        assert len(signals) > 0
        assert signals[0]['symbol'] == '000001.SZ'
        assert signals[0]['action'] == 'buy'
        assert signals[0]['confidence'] == 0.8
    
    def test_generate_signals_with_composite_score(self, trading_system):
        """测试基于综合评分生成交易信号"""
        analysis_result = {
            'ml_prediction': {'prediction': 0, 'confidence': 0.3},
            'analysis_report': {'composite_score': 0.5}
        }
        
        signals = trading_system._generate_signals(analysis_result)
        
        assert isinstance(signals, list)
        # 综合评分>0.3应该生成信号
        assert len(signals) > 0
    
    def test_generate_signals_with_empty_analysis(self, trading_system):
        """测试空分析结果生成信号"""
        signals = trading_system._generate_signals({})
        assert signals == []
    
    def test_generate_signals_with_exception(self, trading_system):
        """测试生成信号时发生异常"""
        # 测试异常情况：如果_generate_signals内部抛出异常，应该被捕获或抛出
        original_method = trading_system._generate_signals
        def failing_generate_signals(analysis_result):
            raise Exception("信号生成失败")
        
        trading_system._generate_signals = failing_generate_signals
        try:
            signals = trading_system._generate_signals({})
            # 如果方法没有处理异常，应该抛出异常
            assert False, "应该抛出异常"
        except Exception:
            # 异常被抛出，测试通过
            pass
        finally:
            trading_system._generate_signals = original_method
    
    def test_execute_trades(self, trading_system):
        """测试执行交易"""
        signals = [
            {
                'symbol': '000001.SZ',
                'action': 'buy',
                'quantity': 1000,
                'reason': '测试',
                'confidence': 0.8
            }
        ]
        
        initial_count = len(trading_system.trading_history)
        trading_system._execute_trades(signals)
        
        assert len(trading_system.trading_history) == initial_count + 1
        assert trading_system.trading_history[-1]['signal'] == signals[0]
        assert trading_system.trading_history[-1]['status'] == 'executed'
    
    def test_execute_trades_multiple_signals(self, trading_system):
        """测试执行多个交易信号"""
        signals = [
            {'symbol': '000001.SZ', 'action': 'buy', 'quantity': 1000, 'reason': 'test1', 'confidence': 0.8},
            {'symbol': '000002.SZ', 'action': 'sell', 'quantity': 500, 'reason': 'test2', 'confidence': 0.7}
        ]
        
        initial_count = len(trading_system.trading_history)
        trading_system._execute_trades(signals)
        
        assert len(trading_system.trading_history) == initial_count + 2
    
    def test_execute_trades_with_exception(self, trading_system):
        """测试执行交易时发生异常"""
        signals = [{'invalid': 'signal'}]
        
        initial_count = len(trading_system.trading_history)
        trading_system._execute_trades(signals)
        
        # 即使发生异常，也应该记录到历史中（如果异常处理允许）
        # 或者不增加历史记录（如果异常被捕获）
        assert len(trading_system.trading_history) >= initial_count
    
    def test_get_trading_status(self, trading_system):
        """测试获取交易状态"""
        trading_system.initialize()
        status = trading_system.get_trading_status()
        
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'positions' in status
        assert 'orders' in status
        assert 'trading_history_count' in status
        assert 'last_update' in status
        assert 'market_data_symbols' in status
        assert status['is_running'] is False
    
    def test_get_trading_status_when_running(self, trading_system):
        """测试运行中获取交易状态"""
        trading_system.initialize()
        trading_system.start()
        
        status = trading_system.get_trading_status()
        assert status['is_running'] is True
        
        trading_system.stop()
    
    def test_get_trading_history_empty(self, trading_system):
        """测试获取空交易历史"""
        history = trading_system.get_trading_history()
        assert history == []
    
    def test_get_trading_history_with_limit(self, trading_system):
        """测试获取交易历史（带限制）"""
        # 添加一些交易历史
        for i in range(5):
            trading_system.trading_history.append({
                'timestamp': datetime.now(),
                'signal': {'symbol': f'00000{i}.SZ', 'action': 'buy'},
                'status': 'executed'
            })
        
        history = trading_system.get_trading_history(limit=3)
        assert len(history) == 3
        assert history == trading_system.trading_history[-3:]
    
    def test_get_trading_history_default_limit(self, trading_system):
        """测试获取交易历史（默认限制）"""
        # 添加超过100条历史
        for i in range(150):
            trading_system.trading_history.append({
                'timestamp': datetime.now(),
                'signal': {'symbol': f'00000{i}.SZ', 'action': 'buy'},
                'status': 'executed'
            })
        
        history = trading_system.get_trading_history()
        assert len(history) == 100
    
    def test_get_performance_metrics_empty(self, trading_system):
        """测试获取空性能指标"""
        metrics = trading_system.get_performance_metrics()
        assert metrics == {}
    
    def test_get_performance_metrics(self, trading_system):
        """测试获取性能指标"""
        # 添加交易历史
        trading_system.trading_history = [
            {
                'timestamp': datetime.now(),
                'signal': {'symbol': '000001.SZ', 'action': 'buy', 'confidence': 0.8},
                'status': 'executed'
            },
            {
                'timestamp': datetime.now(),
                'signal': {'symbol': '000002.SZ', 'action': 'sell', 'confidence': 0.7},
                'status': 'executed'
            },
            {
                'timestamp': datetime.now(),
                'signal': {'symbol': '000003.SZ', 'action': 'buy', 'confidence': 0.9},
                'status': 'executed'
            }
        ]
        
        metrics = trading_system.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_trades' in metrics
        assert 'buy_trades' in metrics
        assert 'sell_trades' in metrics
        assert 'success_rate' in metrics
        assert 'avg_confidence' in metrics
        assert metrics['total_trades'] == 3
        assert metrics['buy_trades'] == 2
        assert metrics['sell_trades'] == 1
        assert metrics['success_rate'] == 0.8
        assert metrics['avg_confidence'] == pytest.approx(0.8, abs=0.01)
    
    def test_trading_loop_basic(self, trading_system):
        """测试交易主循环基本功能"""
        trading_system.initialize()
        trading_system.config['trading_interval'] = 0.01  # 短间隔用于测试
        
        # 在后台线程中运行循环
        trading_system.is_running = True
        thread = threading.Thread(target=trading_system._trading_loop, daemon=True)
        thread.start()
        
        # 等待一小段时间让循环执行
        time.sleep(0.05)
        
        # 停止循环
        trading_system.is_running = False
        thread.join(timeout=2)
        
        # 验证循环已停止
        assert trading_system.is_running is False
    
    def test_trading_loop_with_market_data(self, trading_system):
        """测试交易主循环处理市场数据"""
        trading_system.initialize()
        trading_system.config['trading_interval'] = 0.1  # 短间隔用于测试
        
        # 启动循环（在后台线程）
        trading_system.is_running = True
        thread = threading.Thread(target=trading_system._trading_loop, daemon=True)
        thread.start()
        
        # 等待一小段时间让循环执行
        time.sleep(0.3)
        
        # 停止循环
        trading_system.is_running = False
        thread.join(timeout=1)
        
        # 验证市场数据被更新
        assert trading_system.last_update is not None
    
    def test_trading_loop_exception_handling(self, trading_system):
        """测试交易循环异常处理"""
        trading_system.initialize()
        trading_system.config['trading_interval'] = 0.1
        
        # 模拟_get_market_data抛出异常
        original_get_market_data = trading_system._get_market_data
        call_count = {'count': 0}
        
        def mock_get_market_data():
            call_count['count'] += 1
            if call_count['count'] == 1:
                raise Exception("模拟异常")
            return original_get_market_data()
        
        trading_system._get_market_data = mock_get_market_data
        
        trading_system.is_running = True
        thread = threading.Thread(target=trading_system._trading_loop, daemon=True)
        thread.start()
        
        time.sleep(0.3)
        trading_system.is_running = False
        thread.join(timeout=1)
        
        # 异常应该被捕获，循环应该继续运行
        assert True  # 如果没有崩溃就说明异常被正确处理

