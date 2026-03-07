# 修复 backtest_engine 导入问题并进行单元测试计划

## 任务目标
参考网关层和策略服务层架构设计文档，继续深入全面检查并验证 backtest_engine 导入问题，并进行单元测试以验证导入问题已解决。

## 架构设计参考

### 网关层架构 (docs/architecture/gateway_layer_architecture_design.md)
- **核心定位**: 统一入口层，负责API路由管理、负载均衡、安全认证
- **设计原则**: 统一入口、路由管理、负载均衡、安全认证、流量控制、监控告警
- **Web服务**: `src/gateway/web/` 目录包含18个文件

### 策略服务层架构 (docs/architecture/strategy_layer_architecture_design.md)
- **核心组件**: 策略核心、策略智能、策略回测
- **回测引擎**: `src/strategy/backtest/backtest_engine.py`
- **设计原则**: 回测驱动、模块化扩展、性能优化

## 当前问题分析

### 已尝试的修复方案
1. ✗ 直接导入 - 失败
2. ✗ 使用 `importlib.import_module` - 可能仍有问题
3. ✗ 使用 `__getattr__` 延迟导入 - 可能仍有问题

### 需要深入检查的内容
- [ ] 检查 `backtest_engine.py` 文件是否存在语法错误
- [ ] 检查 `backtest_engine.py` 的依赖导入是否正常
- [ ] 在单元测试中验证所有导入方式
- [ ] 检查后台线程与主线程的环境差异
- [ ] 验证 Docker 容器内的文件权限和路径

## 检查步骤

### 第一阶段：代码质量检查（10分钟）
1. 检查 `backtest_engine.py` 文件语法
2. 检查文件编码和换行符
3. 检查文件权限
4. 验证文件内容完整性

### 第二阶段：依赖分析（10分钟）
1. 分析 `backtest_engine.py` 的所有导入依赖
2. 检查依赖模块是否存在
3. 检查是否存在循环导入
4. 验证依赖模块的加载顺序

### 第三阶段：单元测试（20分钟）
1. 编写单元测试脚本测试导入
2. 测试多种导入方式
3. 测试后台线程导入
4. 验证导入后的功能正常

### 第四阶段：集成测试（10分钟）
1. 在策略优化服务中测试导入
2. 验证参数优化功能
3. 检查日志输出
4. 进行端到端测试

## 单元测试计划

### 测试脚本 1: 基础导入测试
```python
def test_basic_import():
    """测试基础导入"""
    try:
        from src.strategy.backtest.backtest_engine import BacktestEngine
        assert BacktestEngine is not None
        print("✓ 基础导入测试通过")
        return True
    except Exception as e:
        print(f"✗ 基础导入测试失败: {e}")
        return False
```

### 测试脚本 2: 包导入测试
```python
def test_package_import():
    """测试包级别导入"""
    try:
        import src.strategy.backtest
        assert hasattr(src.strategy.backtest, 'BacktestEngine')
        print("✓ 包导入测试通过")
        return True
    except Exception as e:
        print(f"✗ 包导入测试失败: {e}")
        return False
```

### 测试脚本 3: 后台线程导入测试
```python
def test_thread_import():
    """测试后台线程导入"""
    import threading
    result = []
    
    def import_in_thread():
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine
            result.append(True)
        except Exception as e:
            result.append(False)
    
    thread = threading.Thread(target=import_in_thread)
    thread.start()
    thread.join()
    
    if result and result[0]:
        print("✓ 后台线程导入测试通过")
        return True
    else:
        print("✗ 后台线程导入测试失败")
        return False
```

### 测试脚本 4: 功能测试
```python
def test_functionality():
    """测试导入后的功能"""
    try:
        from src.strategy.backtest.backtest_engine import BacktestEngine
        engine = BacktestEngine()
        assert engine is not None
        print("✓ 功能测试通过")
        return True
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False
```

## 预期结果
- 所有单元测试通过
- 策略优化服务正常运行
- 无 `No module named 'backtest_engine'` 错误
- 参数优化功能可用

## 风险与缓解
| 风险 | 缓解措施 |
|------|----------|
| 代码质量问题 | 使用语法检查工具 |
| 依赖问题 | 全面分析依赖链 |
| 环境问题 | 在容器中测试 |
| 线程问题 | 专门测试后台线程 |
