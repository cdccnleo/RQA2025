# RQA 服务层使用示例

## 概述

本文档提供了RQA服务层的核心使用示例。

## 基础服务使用

### 自定义服务示例

```python
from src.services.base_service import BaseService, ServiceStatus
import time

class CustomDataService(BaseService):
    def __init__(self, data_source: str):
        super().__init__()
        self.data_source = data_source
        self.data_cache = {}
        self.start_time = None
    
    def _start(self) -> bool:
        try:
            print(f"连接到数据源: {self.data_source}")
            self.start_time = time.time()
            self.data_cache = {"initial": "data"}
            return True
        except Exception as e:
            print(f"启动失败: {e}")
            return False
    
    def _stop(self) -> bool:
        try:
            self.data_cache.clear()
            return True
        except Exception as e:
            print(f"停止失败: {e}")
            return False
    
    def get_data(self, key: str):
        return self.data_cache.get(key)
    
    def set_data(self, key: str, value):
        try:
            self.data_cache[key] = value
            return True
        except Exception:
            return False
    
    def health_check(self):
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            "status": "healthy" if self.get_status() == ServiceStatus.RUNNING else "unhealthy",
            "uptime": uptime,
            "cache_size": len(self.data_cache),
            "data_source": self.data_source
        }

# 使用示例
def demo_custom_service():
    data_service = CustomDataService("mysql://localhost:3306/rqa")
    
    if data_service.start():
        print("✅ 服务启动成功")
        
        # 设置和获取数据
        data_service.set_data("user:123", {"name": "张三", "age": 30})
        user_data = data_service.get_data("user:123")
        print(f"用户数据: {user_data}")
        
        # 健康检查
        health = data_service.health_check()
        print(f"健康状态: {health}")
        
        data_service.stop()
        print("✅ 服务停止成功")
    else:
        print("❌ 服务启动失败")
```

## 模型服务使用

### 基础预测示例

```python
from src.services.model_service import ModelService

def demo_model_prediction():
    model_service = ModelService("models/stock_predictor.joblib")
    
    if model_service.start():
        print("✅ 模型服务启动成功")
        
        # 执行预测
        test_data = {
            "features": [1.2, 3.4, 5.6, 7.8, 9.0],
            "timestamp": "2025-08-04T10:00:00Z",
            "symbol": "AAPL"
        }
        
        result = model_service.predict(test_data)
        print(f"预测结果: {result.get('prediction', 'N/A')}")
        print(f"置信度: {result.get('confidence', 'N/A')}")
        
        # 获取模型信息
        model_info = model_service.get_model_info()
        print(f"模型类型: {model_info.get('model_type', 'N/A')}")
        
        model_service.stop()
        print("✅ 模型服务停止成功")
    else:
        print("❌ 模型服务启动失败")
```

### A/B测试示例

```python
def demo_ab_test():
    model_service = ModelService("models/stock_predictor.joblib", enable_ab_test=True)
    
    if model_service.start():
        test_data = {
            "features": [1.2, 3.4, 5.6, 7.8, 9.0],
            "symbol": "AAPL"
        }
        
        # A组测试
        result_a = model_service.ab_test_predict(test_data, "A")
        print(f"A组结果: {result_a.get('prediction', 'N/A')}")
        
        # B组测试
        result_b = model_service.ab_test_predict(test_data, "B")
        print(f"B组结果: {result_b.get('prediction', 'N/A')}")
        
        model_service.stop()
```

## 微服务使用

### 服务注册与发现

```python
from src.services.micro_service import MicroService, ServiceType

def demo_microservice():
    # 创建微服务
    api_service = MicroService("api-service", ServiceType.API, port=8001)
    model_service = MicroService("model-service", ServiceType.MODEL, port=8002)
    
    # 启动服务
    services = [api_service, model_service]
    for service in services:
        if service.start():
            print(f"✅ {service.service_name} 启动成功")
    
    # 注册服务
    for service in services:
        if service.register_service():
            print(f"✅ {service.service_name} 注册成功")
    
    # 服务发现
    for service in services:
        for target_service in services:
            if target_service.service_name != service.service_name:
                info = service.discover_service(target_service.service_name)
                if info:
                    print(f"{service.service_name} 发现 {info['name']}")
    
    # 停止服务
    for service in services:
        service.stop()
        print(f"✅ {service.service_name} 停止成功")
```

## 缓存服务使用

### 基础缓存操作

```python
from src.services.cache_service import CacheService

def demo_caching():
    cache_service = CacheService(max_size=1000, ttl=3600)
    
    if cache_service.start():
        print("✅ 缓存服务启动成功")
        
        # 设置缓存
        cache_service.set("user:123", {"name": "张三", "age": 30})
        cache_service.set("config:app", {"version": "1.0.0", "debug": True})
        
        # 获取缓存
        user_data = cache_service.get("user:123")
        config_data = cache_service.get("config:app")
        print(f"用户数据: {user_data}")
        print(f"配置数据: {config_data}")
        
        # 缓存统计
        stats = cache_service.get_cache_stats()
        print(f"缓存条目数: {stats.get('size', 0)}")
        print(f"命中率: {stats.get('hit_rate', 0):.2%}")
        
        cache_service.stop()
        print("✅ 缓存服务停止成功")
    else:
        print("❌ 缓存服务启动失败")
```

## 智能监控使用

### 基础监控

```python
from scripts.services.intelligent_monitoring import IntelligentMonitoring

def demo_monitoring():
    monitoring = IntelligentMonitoring()
    
    if monitoring.start_monitoring():
        print("✅ 智能监控启动成功")
        
        # 收集指标
        metrics = monitoring.collect_metrics()
        print(f"CPU使用率: {metrics.get('cpu_usage', 0):.2f}%")
        print(f"内存使用率: {metrics.get('memory_usage', 0):.2f}%")
        print(f"响应时间: {metrics.get('response_time', 0):.2f}ms")
        
        # 检查告警
        alerts = monitoring.check_alerts()
        if alerts:
            print(f"发现 {len(alerts)} 个告警")
            for alert in alerts:
                print(f"  [{alert.get('severity', 'UNKNOWN')}] {alert.get('message', 'N/A')}")
        else:
            print("✅ 无告警")
        
        # 生成报告
        report_path = monitoring.generate_monitoring_report()
        print(f"监控报告已生成: {report_path}")
        
        print("✅ 监控演示完成")
    else:
        print("❌ 智能监控启动失败")
```

## 综合应用示例

### 完整的交易预测系统

```python
class TradingPredictionSystem:
    def __init__(self):
        self.model_service = None
        self.cache_service = None
        self.monitoring = None
        self.is_running = False
    
    def initialize(self):
        self.model_service = ModelService("models/stock_predictor.joblib")
        self.cache_service = CacheService(max_size=2000, ttl=3600)
        
        services = [self.model_service, self.cache_service]
        for service in services:
            if not service.start():
                return False
        
        self.monitoring = IntelligentMonitoring()
        if not self.monitoring.start_monitoring():
            return False
        
        self.is_running = True
        print("✅ 系统初始化完成")
        return True
    
    def predict_stock(self, symbol: str, features: list):
        if not self.is_running:
            return {"error": "系统未运行"}
        
        # 生成缓存键
        cache_key = f"prediction:{symbol}:{hash(str(features))}"
        
        # 尝试从缓存获取
        cached_result = self.cache_service.get(cache_key)
        if cached_result:
            print(f"✅ 缓存命中: {symbol}")
            return cached_result
        
        # 执行预测
        print(f"⏱️  执行预测: {symbol}")
        prediction_data = {
            "features": features,
            "symbol": symbol,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        result = self.model_service.predict(prediction_data)
        result["cache_hit"] = False
        
        # 缓存结果
        self.cache_service.set(cache_key, result, ttl=3600)
        print(f"💾 结果已缓存: {symbol}")
        
        return result
    
    def shutdown(self):
        if self.model_service:
            self.model_service.stop()
        if self.cache_service:
            self.cache_service.stop()
        if self.monitoring:
            report_path = self.monitoring.generate_monitoring_report()
            print(f"✅ 监控报告已生成: {report_path}")
        self.is_running = False
        print("✅ 系统已关闭")

def demo_trading_system():
    system = TradingPredictionSystem()
    
    if system.initialize():
        # 执行预测
        test_predictions = [
            {"symbol": "AAPL", "features": [1.1, 2.2, 3.3, 4.4, 5.5]},
            {"symbol": "GOOGL", "features": [1.2, 2.3, 3.4, 4.5, 5.6]},
            {"symbol": "MSFT", "features": [1.3, 2.4, 3.5, 4.6, 5.7]}
        ]
        
        for pred in test_predictions:
            result = system.predict_stock(pred["symbol"], pred["features"])
            prediction = result.get("prediction", "N/A")
            cache_hit = result.get("cache_hit", False)
            print(f"{pred['symbol']}: 预测={prediction}, 缓存={cache_hit}")
        
        system.shutdown()
        print("✅ 演示完成")
    else:
        print("❌ 系统初始化失败")

if __name__ == "__main__":
    demo_trading_system()
```

## 最佳实践

1. **服务初始化**: 始终检查服务启动状态
2. **错误处理**: 使用try-catch包装关键操作
3. **资源清理**: 确保在程序结束时停止所有服务
4. **缓存策略**: 合理设置缓存TTL和大小
5. **监控集成**: 在生产环境中启用监控
6. **健康检查**: 定期检查服务健康状态

## 总结

这些示例展示了RQA服务层的核心功能使用方法，为开发者提供了实用的参考。 