# 基础设施层质量检查工具

自动化质量检查工具，为RQA2025基础设施层提供全面的质量保障。

## 功能特性

### 🔍 代码重复检测
- **精确匹配**: 检测完全相同的代码块
- **相似度分析**: 基于AST的相似代码检测
- **智能过滤**: 自动过滤注释、import语句等
- **阈值控制**: 可配置重复次数和相似度阈值

### 🔗 接口一致性检查
- **抽象方法验证**: 确保接口方法正确实现
- **方法签名检查**: 验证参数和返回值一致性
- **继承关系验证**: 检查类继承体系的正确性
- **自动发现**: 智能识别接口实现关系

### 📊 代码复杂度分析
- **圈复杂度计算**: McCabe复杂度指标
- **可维护性指数**: 综合代码质量评估
- **嵌套深度检查**: 控制代码结构复杂度
- **函数长度监控**: 防止函数过度膨胀

## 安装使用

### 环境要求
- Python 3.8+
- 无额外依赖（使用标准库）

### 基本用法

```bash
# 检查当前目录
python -m tools.quality_check .

# 检查特定目录
python -m tools.quality_check src/infrastructure/cache

# 使用基础设施专用配置
python -m tools.quality_check --config infrastructure .

# 生成所有报告格式
python -m tools.quality_check --reports all .

# 仅运行特定检查器
python -m tools.quality_check --checkers duplicate complexity .

# 详细输出模式
python -m tools.quality_check --verbose .
```

### 高级用法

```bash
# 使用自定义配置文件
python -m tools.quality_check --config-file my_config.json .

# 指定输出目录
python -m tools.quality_check --output-dir reports .

# 禁用控制台颜色
python -m tools.quality_check --no-color .

# CI/CD集成（失败时退出码非0）
python -m tools.quality_check .
```

## 配置说明

### 默认配置

工具提供两种预设配置：

- **default**: 通用Python项目配置
- **infrastructure**: 基础设施层专用配置（更严格的标准）

### 自定义配置

创建JSON配置文件：

```json
{
  "enabled_checkers": ["duplicate", "interface", "complexity"],
  "fail_on_error": true,
  "fail_on_critical": true,
  "duplicate": {
    "min_lines": 5,
    "similarity_threshold": 0.8,
    "duplicate_threshold": 3
  },
  "complexity": {
    "max_cyclomatic_complexity": 10,
    "max_lines_per_function": 50
  },
  "reporters": {
    "console": {"enabled": true, "colors": true},
    "json": {"enabled": true, "output_file": "report.json"},
    "html": {"enabled": true, "output_file": "report.html"}
  }
}
```

## 检查器详情

### 代码重复检查器 (duplicate)

检测代码中的重复模式：

```python
# 配置选项
{
  "min_lines": 5,              # 最少行数
  "max_lines": 50,             # 最大行数
  "similarity_threshold": 0.8, # 相似度阈值
  "duplicate_threshold": 3,    # 重复次数阈值
  "ignore_imports": true,      # 忽略import
  "ignore_comments": true,     # 忽略注释
  "ignore_docstrings": true    # 忽略文档字符串
}
```

### 接口一致性检查器 (interface)

检查接口实现正确性：

```python
# 配置选项
{
  "check_abstract_methods": true,     # 检查抽象方法
  "check_method_signatures": true,    # 检查方法签名
  "check_property_implementations": true,  # 检查属性
  "allow_extra_methods": true,        # 允许额外方法
  "strict_mode": false               # 严格模式
}
```

### 复杂度检查器 (complexity)

分析代码复杂度指标：

```python
# 配置选项
{
  "max_cyclomatic_complexity": 10,    # 最大圈复杂度
  "max_lines_per_function": 50,       # 函数最大行数
  "max_nesting_depth": 4,            # 最大嵌套深度
  "max_parameters": 5,               # 最大参数数量
  "min_maintainability_index": 50,   # 最小可维护性指数
  "check_functions": true,           # 检查函数
  "check_classes": true,             # 检查类
  "check_modules": true             # 检查模块
}
```

## 报告格式

### 控制台报告
实时显示检查结果，支持彩色输出和详细模式。

### JSON报告
结构化数据格式，便于程序处理和CI/CD集成。

```json
{
  "summary": {
    "total_issues": 25,
    "total_errors": 3,
    "status": "warning"
  },
  "results": {
    "duplicate": {...},
    "interface": {...},
    "complexity": {...}
  }
}
```

### HTML报告
美观的Web页面报告，包含图表和详细分析。

## CI/CD集成

### GitHub Actions示例

```yaml
name: Quality Check
on: [push, pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Run Quality Check
      run: python -m tools.quality_check --config infrastructure .
    - name: Upload reports
      uses: actions/upload-artifact@v2
      with:
        name: quality-reports
        path: |
          quality_report.html
          quality_report.json
```

### 其他CI系统

```bash
# Jenkins
python -m tools.quality_check --reports json --output-dir reports .

# GitLab CI
python -m tools.quality_check --config infrastructure --reports all .

# Azure DevOps
python -m tools.quality_check --no-color --reports json .
```

## 扩展开发

### 添加新的检查器

1. 继承`BaseChecker`类：

```python
from ..core.base_checker import BaseChecker

class CustomChecker(BaseChecker):
    @property
    def checker_name(self) -> str:
        return "custom_checker"

    @property
    def checker_description(self) -> str:
        return "自定义检查器"

    def check(self, target_path: str):
        result = self._create_result()
        # 实现检查逻辑
        return result
```

2. 在`QualityChecker`中注册：

```python
from .checkers.custom_checker import CustomChecker

# 在_initialize_checkers中添加
checkers['custom'] = CustomChecker(config.get('custom', {}))
```

### 添加新的报告器

1. 实现报告器接口：

```python
class CustomReporter:
    def __init__(self, config=None):
        self.config = config or {}

    def report(self, results):
        # 生成报告逻辑
        return report_content
```

2. 在主程序中集成：

```python
from .reporters.custom_reporter import CustomReporter

# 在generate_reports中添加
if reporters_config.get('custom', {}).get('enabled', False):
    reporter = CustomReporter(reporters_config['custom'])
    reporter.report(results)
```

## 故障排除

### 常见问题

1. **ImportError**: 检查Python路径和模块结构
2. **PermissionError**: 确保对目标目录有读取权限
3. **MemoryError**: 减少检查的文件数量或增加内存
4. **Timeout**: 调整检查器超时设置

### 调试模式

```bash
# 启用详细日志
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python -m tools.quality_check --verbose .

# 检查特定文件
python -m tools.quality_check --checkers complexity path/to/file.py
```

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看[LICENSE](../LICENSE)文件了解详情。

## 联系方式

- 项目维护者: 专项修复小组
- 问题反馈: [GitHub Issues](../../issues)
- 文档更新: [Wiki](../../wiki)
