"""
策略执行API集成测试
测试策略执行相关的所有API端点
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import time

# 导入路由
from src.gateway.web.strategy_execution_routes import router


class TestStrategyExecutionAPI(unittest.TestCase):
    """
    策略执行API集成测试类
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        # 创建FastAPI应用并添加路由
        self.app = FastAPI()
        self.app.include_router(router)
        
        # 创建测试客户端
        self.client = TestClient(self.app)
    
    def test_get_strategy_execution_status(self):
        """
        测试获取策略执行状态API
        """
        # 模拟策略执行服务返回数据
        mock_status = {
            "strategies": [
                {
                    "id": "test_strategy",
                    "name": "Test Strategy",
                    "type": "trend_following",
                    "status": "running",
                    "latency": 10,
                    "throughput": 100,
                    "signals_count": 50,
                    "positions_count": 5
                }
            ],
            "running_count": 1,
            "paused_count": 0,
            "stopped_count": 0,
            "total_count": 1
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_strategy_execution_status') as mock_get_status:
            mock_get_status.return_value = mock_status
            
            response = self.client.get("/api/v1/strategy/execution/status")
            
            # 验证响应
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), mock_status)
    
    def test_get_strategy_execution_status_error(self):
        """
        测试获取策略执行状态API失败的情况
        """
        with patch('src.gateway.web.strategy_execution_service.get_strategy_execution_status') as mock_get_status:
            mock_get_status.side_effect = Exception("Test error")
            
            response = self.client.get("/api/v1/strategy/execution/status")
            
            # 验证响应
            self.assertEqual(response.status_code, 500)
            self.assertIn("获取失败", response.json()["detail"])
    
    def test_get_strategy_execution_metrics(self):
        """
        测试获取策略执行性能指标API
        """
        # 模拟策略执行服务返回数据
        mock_metrics = {
            "avg_latency": 5,
            "today_signals": 100,
            "total_trades": 50,
            "latency_history": [],
            "throughput_history": []
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics
            
            response = self.client.get("/api/v1/strategy/execution/metrics")
            
            # 验证响应
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), mock_metrics)
    
    def test_get_strategy_execution_metrics_error(self):
        """
        测试获取策略执行性能指标API失败的情况
        """
        with patch('src.gateway.web.strategy_execution_service.get_execution_metrics') as mock_get_metrics:
            mock_get_metrics.side_effect = Exception("Test error")
            
            response = self.client.get("/api/v1/strategy/execution/metrics")
            
            # 验证响应
            self.assertEqual(response.status_code, 500)
            self.assertIn("获取失败", response.json()["detail"])
    
    def test_start_strategy_execution_success(self):
        """
        测试启动策略执行API成功的情况
        """
        strategy_id = "test_strategy"
        
        # 模拟策略执行服务返回成功
        with patch('src.gateway.web.strategy_execution_service.start_strategy') as mock_start_strategy:
            mock_start_strategy.return_value = True
            
            # 模拟事件总线
            with patch('src.gateway.web.strategy_execution_routes._get_event_bus') as mock_get_event_bus:
                mock_event_bus = Mock()
                mock_get_event_bus.return_value = mock_event_bus
                
                # 模拟WebSocket管理器
                with patch('src.gateway.web.strategy_execution_routes._get_websocket_manager') as mock_get_websocket:
                    mock_websocket = Mock()
                    mock_websocket.broadcast = Mock()
                    mock_get_websocket.return_value = mock_websocket
                    
                    # 模拟业务流程编排器
                    with patch('src.gateway.web.strategy_execution_routes._get_orchestrator') as mock_get_orchestrator:
                        mock_orchestrator = Mock()
                        mock_orchestrator.start_process = Mock(return_value="test_process_id")
                        mock_get_orchestrator.return_value = mock_orchestrator
                        
                        response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/start")
                        
                        # 验证响应
                        self.assertEqual(response.status_code, 200)
                        self.assertTrue(response.json()["success"])
                        self.assertEqual(response.json()["strategy_id"], strategy_id)
                        self.assertIn(f"策略 {strategy_id} 已启动", response.json()["message"])
    
    def test_start_strategy_execution_not_found(self):
        """
        测试启动策略执行API策略不存在的情况
        """
        strategy_id = "non_existent_strategy"
        
        # 模拟策略执行服务返回失败
        with patch('src.gateway.web.strategy_execution_service.start_strategy') as mock_start_strategy:
            mock_start_strategy.return_value = False
            
            response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/start")
            
            # 验证响应
            self.assertEqual(response.status_code, 404)
            self.assertIn(f"策略 {strategy_id} 不存在或启动失败", response.json()["detail"])
    
    def test_start_strategy_execution_error(self):
        """
        测试启动策略执行API失败的情况
        """
        strategy_id = "test_strategy"
        
        # 模拟策略执行服务抛出异常
        with patch('src.gateway.web.strategy_execution_service.start_strategy') as mock_start_strategy:
            mock_start_strategy.side_effect = Exception("Test error")
            
            response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/start")
            
            # 验证响应
            self.assertEqual(response.status_code, 500)
            self.assertIn("启动失败", response.json()["detail"])
    
    def test_pause_strategy_execution_success(self):
        """
        测试暂停策略执行API成功的情况
        """
        strategy_id = "test_strategy"
        
        # 模拟策略执行服务返回成功
        with patch('src.gateway.web.strategy_execution_service.pause_strategy') as mock_pause_strategy:
            mock_pause_strategy.return_value = True
            
            # 模拟事件总线
            with patch('src.gateway.web.strategy_execution_routes._get_event_bus') as mock_get_event_bus:
                mock_event_bus = Mock()
                mock_get_event_bus.return_value = mock_event_bus
                
                # 模拟WebSocket管理器
                with patch('src.gateway.web.strategy_execution_routes._get_websocket_manager') as mock_get_websocket:
                    mock_websocket = Mock()
                    mock_websocket.broadcast = Mock()
                    mock_get_websocket.return_value = mock_websocket
                    
                    # 模拟业务流程编排器
                    with patch('src.gateway.web.strategy_execution_routes._get_orchestrator') as mock_get_orchestrator:
                        mock_orchestrator = Mock()
                        mock_orchestrator.update_process_state = Mock()
                        mock_get_orchestrator.return_value = mock_orchestrator
                        
                        response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/pause")
                        
                        # 验证响应
                        self.assertEqual(response.status_code, 200)
                        self.assertTrue(response.json()["success"])
                        self.assertEqual(response.json()["strategy_id"], strategy_id)
                        self.assertIn(f"策略 {strategy_id} 已暂停", response.json()["message"])
    
    def test_pause_strategy_execution_not_found(self):
        """
        测试暂停策略执行API策略不存在的情况
        """
        strategy_id = "non_existent_strategy"
        
        # 模拟策略执行服务返回失败
        with patch('src.gateway.web.strategy_execution_service.pause_strategy') as mock_pause_strategy:
            mock_pause_strategy.return_value = False
            
            response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/pause")
            
            # 验证响应
            self.assertEqual(response.status_code, 404)
            self.assertIn(f"策略 {strategy_id} 不存在", response.json()["detail"])
    
    def test_pause_strategy_execution_error(self):
        """
        测试暂停策略执行API失败的情况
        """
        strategy_id = "test_strategy"
        
        # 模拟策略执行服务抛出异常
        with patch('src.gateway.web.strategy_execution_service.pause_strategy') as mock_pause_strategy:
            mock_pause_strategy.side_effect = Exception("Test error")
            
            response = self.client.post(f"/api/v1/strategy/execution/{strategy_id}/pause")
            
            # 验证响应
            self.assertEqual(response.status_code, 500)
            self.assertIn("暂停失败", response.json()["detail"])
    
    def test_get_realtime_metrics(self):
        """
        测试获取实时策略处理指标API
        """
        # 模拟策略执行服务返回数据
        mock_metrics = {
            "metrics": {
                "processing_latency": 5
            },
            "stream_metrics": {
                "processing_latency": 5
            },
            "strategies": [
                {
                    "id": "test_strategy",
                    "name": "Test Strategy",
                    "type": "trend_following",
                    "metrics": {
                        "latency": 10,
                        "signals_count": 50,
                        "positions_count": 5,
                        "trades_count": 20
                    }
                }
            ],
            "history": {
                "latency": [],
                "throughput": []
            }
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics
            
            response = self.client.get("/api/v1/strategy/realtime/metrics")
            
            # 验证响应
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), mock_metrics)
    
    def test_get_realtime_metrics_error(self):
        """
        测试获取实时策略处理指标API失败的情况
        """
        with patch('src.gateway.web.strategy_execution_service.get_realtime_metrics') as mock_get_metrics:
            mock_get_metrics.side_effect = Exception("Test error")
            
            response = self.client.get("/api/v1/strategy/realtime/metrics")
            
            # 验证响应
            self.assertEqual(response.status_code, 500)
            self.assertIn("获取失败", response.json()["detail"])
    
    def test_get_realtime_signals(self):
        """
        测试获取最近信号API
        """
        # 模拟交易信号服务返回数据
        mock_signals = [
            {
                "id": "signal_1",
                "strategy_id": "test_strategy",
                "strategy_name": "Test Strategy",
                "symbol": "AAPL",
                "type": "buy",
                "timestamp": time.time()
            }
        ]
        
        with patch('src.gateway.web.trading_signal_service.get_realtime_signals') as mock_get_signals:
            mock_get_signals.return_value = mock_signals
            
            response = self.client.get("/api/v1/strategy/realtime/signals")
            
            # 验证响应
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["signals"]), 1)
            self.assertEqual(response.json()["signals"][0]["symbol"], "AAPL")
    
    def test_get_realtime_signals_empty(self):
        """
        测试获取最近信号API返回空列表的情况
        """
        with patch('src.gateway.web.trading_signal_service.get_realtime_signals') as mock_get_signals:
            mock_get_signals.return_value = []
            
            # 模拟实时引擎
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
                mock_engine = Mock()
                mock_engine.strategies = {}
                mock_get_engine.return_value = mock_engine
                
                response = self.client.get("/api/v1/strategy/realtime/signals")
                
                # 验证响应
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["signals"]), 0)


if __name__ == '__main__':
    unittest.main()
