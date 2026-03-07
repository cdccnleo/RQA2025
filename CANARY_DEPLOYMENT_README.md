# RQA2025 灰度发布指南

## 📋 概述

RQA2025 采用先进的灰度发布（Canary Deployment）策略，实现零停机、安全的版本更新。通过渐进式部署和实时监控，确保新版本的质量和稳定性。

## 🏗️ 架构设计

### 部署策略
- **金丝雀部署**: 先部署少量新版本，验证后再逐步扩大
- **蓝绿部署**: 同时运行新旧版本，通过负载均衡切换流量
- **滚动更新**: 逐步替换旧版本实例，避免服务中断

### 核心组件
- `scripts/canary_deployment.py` - 灰度发布主脚本
- `scripts/automated_canary_rollout.py` - 自动化发布流程
- `scripts/canary_monitor.py` - 实时监控面板
- `canary_config.json` - 发布配置
- `docker-compose.canary.yml` - 金丝雀环境配置

## 🚀 快速开始

### 1. 准备环境
```bash
# 确保Docker环境正常运行
docker-compose -f docker-compose.prod.yml ps

# 检查配置文件
cat canary_config.json
```

### 2. 执行自动化灰度发布
```bash
# 预览发布计划
python scripts/automated_canary_rollout.py --version v1.2.3 --dry-run

# 执行完整发布
python scripts/automated_canary_rollout.py --version v1.2.3
```

### 3. 手动控制发布流程
```bash
# 构建镜像
python scripts/canary_deployment.py build --version v1.2.3

# 金丝雀部署 (10%)
python scripts/canary_deployment.py canary --version v1.2.3 --percentage 10

# 监控部署状态
python scripts/canary_monitor.py --version v1.2.3 --duration 300

# 全量发布
python scripts/canary_deployment.py rollout --version v1.2.3
```

## 📊 监控和告警

### 实时监控面板
```bash
# 启动监控面板
python scripts/canary_monitor.py --version v1.2.3
```

### Grafana仪表板
访问: http://localhost:3000
- 用户名: admin
- 密码: GrafanaAdmin123!
- 仪表板: "RQA2025 灰度发布监控"

### 监控指标
- ✅ 应用健康状态
- ✅ 响应时间对比
- ✅ 错误率统计
- ✅ CPU/内存使用率
- ✅ 流量分布比例
- ✅ 容器运行状态

## ⚙️ 配置说明

### canary_config.json
```json
{
  "strategy": "canary",
  "total_instances": 6,
  "canary_instances": 1,
  "rollout_percentage": [10, 25, 50, 75, 100],
  "health_check_interval": 30,
  "rollback_threshold": 0.1,
  "monitoring": {
    "prometheus_url": "http://localhost:9090",
    "grafana_url": "http://localhost:3000"
  },
  "metrics": {
    "response_time_threshold": 1000,
    "error_rate_threshold": 0.05,
    "cpu_usage_threshold": 80
  }
}
```

### 发布阶段
1. **10%**: 金丝雀测试阶段，监控关键指标
2. **25%**: 小规模验证，检查业务逻辑
3. **50%**: 中等规模测试，验证系统负载
4. **75%**: 大规模预发布，准备全量切换
5. **100%**: 全量发布，完成版本更新

## 🔄 回滚策略

### 自动回滚触发条件
- 错误率超过 5%
- 响应时间超过 1000ms
- CPU 使用率超过 80%
- 应用健康检查失败

### 手动回滚
```bash
# 回滚到指定版本
python scripts/canary_deployment.py rollback --version v1.2.2
```

## 📈 最佳实践

### 发布前检查
- [ ] 代码审查完成
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试完成
- [ ] 安全扫描通过

### 发布中监控
- [ ] 实时监控面板运行
- [ ] 业务指标正常
- [ ] 用户反馈收集
- [ ] 备用方案就绪

### 发布后验证
- [ ] 功能测试通过
- [ ] 性能指标正常
- [ ] 用户无异常反馈
- [ ] 监控告警清除

## 🛠️ 故障排除

### 常见问题

#### 镜像构建失败
```bash
# 检查Docker状态
docker system info

# 清理构建缓存
docker system prune -f

# 重新构建
docker build --no-cache -t rqa2025-app:v1.2.3 .
```

#### 容器启动失败
```bash
# 查看容器日志
docker-compose -f docker-compose.canary.yml logs app-canary

# 检查端口冲突
netstat -tulpn | grep :8000

# 清理失败的容器
docker-compose -f docker-compose.canary.yml down
```

#### 监控数据异常
```bash
# 检查Prometheus状态
curl http://localhost:9090/-/healthy

# 重启监控栈
docker-compose -f docker-compose.prod.yml restart prometheus grafana

# 检查指标配置
cat monitoring/prometheus.canary.yml
```

## 📋 发布历史

### 查看发布记录
```bash
# 查看发布历史
cat rollout_history.json

# 查看监控报告
ls canary_monitor_report_*.json
```

### 发布报告示例
```json
{
  "version": "v1.2.3",
  "timestamp": "2025-10-10T15:30:00Z",
  "status": "success",
  "metrics": {
    "error_rate": 0.02,
    "response_time": 245,
    "cpu_usage": 65,
    "traffic_distribution": {
      "canary": 100,
      "stable": 0
    }
  }
}
```

## 🔧 高级配置

### 自定义监控指标
在 `canary_config.json` 中添加自定义阈值：
```json
"custom_metrics": {
  "business_metric_1": {
    "query": "rate(business_orders_total[5m])",
    "threshold": 100
  }
}
```

### 多环境部署
```bash
# 开发环境
python scripts/canary_deployment.py canary --env dev --version v1.2.3

# 测试环境
python scripts/canary_deployment.py canary --env test --version v1.2.3

# 生产环境
python scripts/canary_deployment.py rollout --env prod --version v1.2.3
```

## 📞 支持和帮助

### 获取帮助
- 查看脚本帮助: `python scripts/canary_deployment.py --help`
- 检查日志: `tail -f logs/canary_deployment.log`
- 监控文档: `monitoring/README.md`

### 联系支持
- 技术支持: dev-support@rqa2025.com
- 紧急联系: emergency@rqa2025.com
- 文档更新: wiki@rqa2025.com

---

**版本**: v1.0.0
**更新日期**: 2025-10-10
**维护者**: RQA2025 DevOps Team


