# RQA2025 基础设施层 v2.0 发布说明

## 版本信息
- **版本号**: 2.0.0
- **发布日期**: 2023-11-15
- **兼容性**: Python 3.8+, PostgreSQL 12+, Redis 6+

## 1. 重大更新

### 1.1 配置管理系统增强
- ✅ 新增配置热更新功能
- ✅ 支持多环境配置隔离
- ✅ 增加配置版本控制
- ⚠️ 变更: 配置文件必须存放在`/config`目录下

### 1.2 监控系统重构
- ✅ 新增系统资源监控(SystemMonitor)
- ✅ 量化策略专用监控(BacktestMonitor)
- ✅ 支持Prometheus格式指标导出
- ⚠️ 废弃: 旧版监控API将在v2.2移除

### 1.3 资源配额管理
- ✅ 策略级CPU/GPU/线程配额
- ✅ 实时资源使用监控
- ✅ 配额超限自动告警

## 2. 新特性详解

### 2.1 配置热更新
```python
# 自动重载变更的配置
config = ConfigManager()
config.start_watcher()  # 开启监听

# 修改配置后自动生效
```

### 2.2 回测监控
```python
monitor = BacktestMonitor()
monitor.record_trade(
    symbol="600000.SH",
    action="BUY",
    price=15.2,
    quantity=1000
)
```

### 2.3 资源配额
```python
quota.set_strategy_quota(
    strategy="high_freq",
    cpu=30,  # 30% CPU上限
    gpu_mem=4096  # 4GB显存上限
)
```

## 3. 升级注意事项

### 3.1 必须操作
1. 备份现有配置
2. 迁移配置到`/config`目录结构
3. 更新监控仪表盘模板

### 3.2 推荐操作
- 为生产环境配置配额限制
- 启用配置验证
- 设置监控数据保留策略

## 4. 已知问题

| 问题描述 | 影响版本 | 临时解决方案 | 修复版本 |
|---------|---------|------------|---------|
| GPU监控在Windows下不可用 | v2.0 | 使用WSL2或Linux | v2.1 |
| 配置热更新偶发死锁 | v2.0 | 重启服务 | v2.0.1 |
| 监控数据时间戳时区问题 | v2.0 | 手动指定时区 | v2.1 |
| 加密配置解密失败 | v2.0 | 检查密钥一致性 | v2.0.1 |
| 加密配置热更新延迟 | v2.0 | 手动触发重载 | v2.1 |

## 5. 废弃声明

以下功能将在后续版本移除：
- `LegacyMonitor`类 (替代方案: `ApplicationMonitor`)
- 配置文件`.ini`格式支持 (仅支持`.yaml`)
- 非配额资源分配方式

## 6. 获取帮助

- 文档中心: [docs.rqa2025.com](https://docs.rqa2025.com)
- 技术支持: infra-support@rqa2025.com
- 紧急问题: +86 400-123-4567

> 注意：升级前请务必测试开发环境！  
> 遇到问题可运行 `diagnose.py` 生成诊断报告。
