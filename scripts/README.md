# RQA2025 质量保障与自动化工具集

## 📋 概述

本工具集提供了RQA2025项目的质量保障、文档自动化和版本管理功能：

- **一致性检查工具**: 自动检查代码与文档的一致性
- **文档同步工具**: 自动化文档与代码的同步更新
- **版本管理工具**: 规范化版本管理和发布流程
- **质量调度器**: 定期执行质量保障任务

## 🏗️ 工具架构

```
scripts/
├── quality_assurance/           # 质量保障
│   ├── consistency_checker.py   # 一致性检查器
│   ├── scheduler.py            # 质量调度器
│   └── scheduler_config.json   # 配置
├── documentation_automation/    # 文档自动化
│   ├── doc_sync.py             # 文档同步器
│   └── templates/              # 模板
└── version_management/          # 版本管理
    └── version_manager.py       # 版本管理器
```

## 🚀 快速开始

### 1. 一致性检查
```bash
# 完整检查
python scripts/quality_assurance/consistency_checker.py

# 快速检查
python scripts/quality_assurance/consistency_checker.py --quick
```

### 2. 文档同步
```bash
# 同步所有文档
python scripts/documentation_automation/doc_sync.py

# 同步指定层
python scripts/documentation_automation/doc_sync.py --layer ml
```

### 3. 版本管理
```bash
# 查看版本
python scripts/version_management/version_manager.py report

# 版本递增
python scripts/version_management/version_manager.py bump --type patch

# 创建发布
python scripts/version_management/version_manager.py release --version 1.1.0
```

### 4. 质量调度器
```bash
# 启动调度器
python scripts/quality_assurance/scheduler.py start

# 手动执行任务
python scripts/quality_assurance/scheduler.py run --task consistency
```

## 📊 报告输出

所有工具报告保存在 `reports/` 目录：
- `reports/technical/consistency/` - 一致性检查报告
- `reports/technical/doc_sync/` - 文档同步报告
- `reports/scheduled/` - 调度任务报告

## 🔧 配置说明

### 调度器配置
编辑 `scripts/quality_assurance/scheduler_config.json` 配置：
- 检查频率 (daily/weekly)
- 执行时间
- 告警通知 (邮件/Slack)
- 质量阈值

### 启用通知
```json
{
  "notification": {
    "enabled": true,
    "email": {
      "smtp_server": "your-smtp.com",
      "username": "your-email@company.com",
      "recipients": ["team@company.com"]
    }
  }
}
```

## 📈 最佳实践

1. **每日检查**: 运行一致性检查确保代码质量
2. **定期同步**: 每周同步文档与代码
3. **版本管理**: 遵循语义化版本规范
4. **自动化监控**: 启用调度器进行自动质量监控

## 🚨 故障排除

### 常见问题
- **权限错误**: 检查文件写权限
- **导入错误**: 确保项目路径正确设置
- **Git错误**: 确认Git环境配置

### 调试模式
```bash
# 启用详细日志
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

---

**RQA2025 质量保障工具集** 🎯🚀✨

*让质量成为习惯，让自动化成为标准*