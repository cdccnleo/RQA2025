# 测试覆盖率面板自动更新指南

本文档介绍如何设置和使用RQA2025项目的测试覆盖率面板自动更新功能。

## 📋 目录

- [快速开始](#快速开始)
- [手动更新](#手动更新)
- [自动更新方式](#自动更新方式)
  - [Linux/macOS环境](#linuxmacos环境)
  - [Windows环境](#windows环境)
  - [CI/CD集成](#cicd集成)
  - [Docker容器](#docker容器)
- [监控和日志](#监控和日志)
- [故障排除](#故障排除)
- [高级配置](#高级配置)

## 🚀 快速开始

### 1. 单次更新面板

```bash
# Linux/macOS
cd /path/to/RQA2025
python scripts/generate_coverage_dashboard.py

# Windows
cd C:\PythonProject\RQA2025
python scripts\generate_coverage_dashboard.py
```

### 2. 查看结果

更新成功后，在浏览器中打开 `reports/coverage_dashboard.html` 查看可视化面板。

## 🔄 手动更新

### 基本用法

```bash
python scripts/generate_coverage_dashboard.py [选项]
```

### 可用选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--db-path PATH` | 数据库路径 | `data/coverage_monitor.db` |
| `--output PATH` | 输出文件路径 | `reports/coverage_dashboard_{timestamp}.html` |
| `--auto-update` | 启用自动定期更新模式 | - |
| `--interval SECONDS` | 自动更新间隔(秒) | 3600 |

### 示例

```bash
# 指定自定义数据库和输出路径
python scripts/generate_coverage_dashboard.py \
    --db-path /path/to/custom.db \
    --output /path/to/dashboard.html

# 启动自动更新模式（每30分钟更新一次）
python scripts/generate_coverage_dashboard.py \
    --auto-update \
    --interval 1800
```

## ⏰ 自动更新方式

### Linux/macOS环境

#### 方法1: 使用自动化脚本

```bash
# 启动后台自动更新服务
./scripts/auto_update_coverage_dashboard.sh --continuous --interval 3600

# 或者使用Python脚本
python scripts/generate_coverage_dashboard.py --auto-update --interval 3600
```

#### 方法2: Cron作业

```bash
# 使用调度脚本设置cron作业（每小时更新）
python scripts/schedule_coverage_dashboard.py --setup-cron 60

# 查看cron状态
python scripts/schedule_coverage_dashboard.py --status

# 移除cron作业
python scripts/schedule_coverage_dashboard.py --remove-cron
```

#### 方法3: systemd服务

```bash
# 生成systemd服务文件
sudo python scripts/schedule_coverage_dashboard.py --generate-systemd

# 启用和启动服务
sudo systemctl daemon-reload
sudo systemctl enable rqa2025-coverage-dashboard
sudo systemctl start rqa2025-coverage-dashboard

# 查看服务状态
sudo systemctl status rqa2025-coverage-dashboard
```

### Windows环境

#### 方法1: 使用批处理脚本

```batch
# 启动后台自动更新（每30分钟）
start /B auto_update_coverage_dashboard.bat --continuous --interval 1800

# 单次更新
auto_update_coverage_dashboard.bat --single
```

#### 方法2: 任务计划程序

```batch
# 设置定时任务（每小时更新）
python scripts\schedule_coverage_dashboard.py --setup-windows-task 60

# 查看任务状态
python scripts\schedule_coverage_dashboard.py --status

# 移除定时任务
python scripts\schedule_coverage_dashboard.py --remove-windows-task
```

### CI/CD集成

项目已配置GitHub Actions工作流，支持以下触发方式：

#### 自动触发
- **代码推送**: 当 `tests/` 或 `src/` 目录有变更时
- **定时任务**: 每天早上8点自动运行
- **手动触发**: 在GitHub界面手动触发

#### 使用方法

1. **推送到主分支**时会自动触发更新
2. **查看结果**:
   - 在Actions标签页查看运行状态
   - 下载artifacts中的覆盖率面板
   - 查看GitHub Pages上的在线版本

#### 本地测试CI/CD

```bash
# 模拟CI/CD环境运行
python scripts/testing/run_tests.py --coverage --output-dir reports/coverage
python scripts/generate_coverage_dashboard.py --output reports/coverage_dashboard.html
```

### Docker容器

#### 构建和运行

```bash
# 构建Docker镜像
docker build -t rqa2025-coverage-dashboard .

# 运行容器（自动更新模式）
docker run -d \
    --name rqa2025-dashboard \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/reports:/app/reports \
    -v $(pwd)/logs:/app/logs \
    rqa2025-coverage-dashboard \
    python scripts/generate_coverage_dashboard.py --auto-update --interval 3600
```

#### Docker Compose配置

```yaml
version: '3.8'
services:
  coverage-dashboard:
    build: .
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
      - ./logs:/app/logs
    command: python scripts/generate_coverage_dashboard.py --auto-update --interval 3600
    restart: unless-stopped
```

## 📊 监控和日志

### 日志文件位置

- **主日志**: `logs/coverage_dashboard_auto_update.log`
- **调度日志**: `logs/coverage_scheduler.log`
- **覆盖率日志**: `logs/coverage_monitor.log`

### 查看日志

```bash
# 查看最新日志
tail -f logs/coverage_dashboard_auto_update.log

# 查看调度状态
python scripts/schedule_coverage_dashboard.py --status

# 查看覆盖率监控日志
tail -n 50 logs/coverage_monitor.log
```

### 监控指标

面板会自动监控以下指标：

- **更新频率**: 每次更新的时间戳
- **成功率**: 更新成功/失败的统计
- **数据新鲜度**: 数据最后更新时间
- **系统状态**: 磁盘空间、内存使用等

## 🔧 故障排除

### 常见问题

#### 1. 数据库不存在

```bash
# 检查数据库文件
ls -la data/coverage_monitor.db

# 如果不存在，使用内置数据生成面板
python scripts/generate_coverage_dashboard.py --db-path /dev/null
```

#### 2. 权限问题

```bash
# Linux/macOS
chmod +x scripts/auto_update_coverage_dashboard.sh
chmod +x scripts/schedule_coverage_dashboard.py

# Windows: 以管理员身份运行
```

#### 3. Python依赖缺失

```bash
pip install plotly pandas
```

#### 4. 端口占用（如果部署到服务器）

```bash
# 检查端口使用情况
netstat -tlnp | grep :80

# 修改Web服务器配置
```

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=/path/to/RQA2025
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from scripts.generate_coverage_dashboard import CoverageDashboardGenerator
gen = CoverageDashboardGenerator()
gen.generate_dashboard()
"
```

## ⚙️ 高级配置

### 自定义数据源

```python
# 修改 generate_coverage_dashboard.py 中的 _get_updated_coverage_data 方法
def _get_updated_coverage_data(self):
    # 从您的数据库或API获取数据
    return {
        'timestamp': '2025-09-13T00:00:00.000000',
        'overall': {
            'coverage': your_coverage_value,
            'statements': your_statements_count,
            'missed': your_missed_count
        },
        'layers': your_layer_data
    }
```

### 自定义样式

```python
# 在 _generate_html_dashboard 方法中修改CSS样式
custom_css = """
    .my-custom-style {
        /* 您的自定义样式 */
    }
"""
```

### 集成到现有系统

```python
# 创建自定义集成脚本
from scripts.generate_coverage_dashboard import CoverageDashboardGenerator

class CustomDashboardIntegrator:
    def __init__(self):
        self.generator = CoverageDashboardGenerator()

    def integrate_with_jenkins(self, build_number, coverage_data):
        # Jenkins集成逻辑
        pass

    def integrate_with_slack(self, webhook_url):
        # Slack通知逻辑
        pass

    def integrate_with_email(self, smtp_config):
        # 邮件通知逻辑
        pass
```

## 📈 最佳实践

### 1. 定期检查

```bash
# 设置每日健康检查
0 9 * * * /path/to/RQA2025/scripts/auto_update_coverage_dashboard.sh --single
```

### 2. 备份策略

```bash
# 备份覆盖率数据
tar -czf coverage_backup_$(date +%Y%m%d).tar.gz data/ reports/ logs/
```

### 3. 性能优化

- **减少更新频率**: 在开发阶段可以使用较低频率
- **压缩存储**: 定期清理旧的覆盖率数据
- **缓存优化**: 使用CDN加速面板访问

### 4. 安全考虑

- **访问控制**: 为面板添加认证
- **数据加密**: 敏感配置使用环境变量
- **审计日志**: 记录所有更新操作

## 🎯 总结

通过以上方法，您可以实现RQA2025项目测试覆盖率面板的完全自动化更新：

1. **立即可用**: 使用单次更新快速生成面板
2. **自动化部署**: 通过cron/Windows任务实现定期更新
3. **CI/CD集成**: 在代码变更时自动更新
4. **容器化部署**: 使用Docker实现可移植的部署
5. **监控告警**: 完整的日志和状态监控

根据您的具体需求选择合适的更新方式，确保测试覆盖率数据始终保持最新和准确。
