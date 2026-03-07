# 智能代码分析器

一个高级的代码质量分析、智能化重构建议生成和代码评审工具，整合多维度分析能力，为RQA2025项目提供全面的代码质量保障。

## 🎯 核心特性

### 🧠 智能化分析
- **全面质量评分**: 基于10+质量指标的综合评分算法
- **AI辅助分析**: 结合AST分析和代码模式识别
- **趋势预测**: 分析代码质量发展趋势和潜在风险
- **智能诊断**: 自动识别代码异味和架构问题

### 📊 多维度指标
- **代码复杂度**: 圈复杂度、可维护性指数、认知复杂度
- **代码质量**: 重复度、测试覆盖率、文档完整性
- **架构健康度**: 耦合度、内聚性、依赖关系分析
- **性能指标**: 内存使用、执行效率、资源消耗

### 🎯 智能重构建议
- **自动化建议**: 基于分析结果生成具体重构方案
- **优先级排序**: 按影响度和工作量智能排序
- **实施指南**: 提供详细的重构步骤和示例代码
- **风险评估**: 评估重构的风险等级和影响范围

### 🔗 工具集成
- **与quality_check集成**: 扩展现有的质量检查功能
- **与smart_duplicate_detector协作**: 整合重复代码检测结果
- **CI/CD集成**: 支持多种CI系统的集成
- **报告聚合**: 统一多种分析工具的报告

## 📁 目录结构

```
tools/smart_code_analyzer/
├── core/                    # 核心数据结构和配置
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── analysis_result.py  # 分析结果定义
│   ├── quality_metrics.py  # 质量指标计算
│   └── refactoring_plan.py # 重构计划
├── analyzers/              # 分析器实现
│   ├── __init__.py
│   ├── ast_analyzer.py     # AST深度分析
│   ├── quality_analyzer.py # 质量分析器
│   ├── dependency_analyzer.py # 依赖分析器
│   └── pattern_analyzer.py # 模式识别器
├── refactorers/            # 重构器
│   ├── __init__.py
│   ├── base_refactorer.py  # 基础重构器
│   ├── method_refactorer.py # 方法重构
│   ├── class_refactorer.py # 类重构
│   └── module_refactorer.py # 模块重构
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── ast_utils.py        # AST工具
│   ├── file_utils.py       # 文件操作工具
│   └── report_utils.py     # 报告生成工具
├── __init__.py             # 包初始化
├── __main__.py             # 命令行入口
└── README.md               # 文档
```

## 🚀 快速开始

### 安装和环境要求

```bash
# 确保在项目根目录
cd /path/to/rqa2025

# 无需额外安装，已集成到项目tools中
python -m tools.smart_code_analyzer --help
```

### 基本使用

```bash
# 分析整个项目
python -m tools.smart_code_analyzer analyze .

# 分析特定模块
python -m tools.smart_code_analyzer analyze src/infrastructure/cache

# 生成质量报告
python -m tools.smart_code_analyzer report --format html .

# 获取重构建议
python -m tools.smart_code_analyzer refactor --top 10 .

# 详细分析模式
python -m tools.smart_code_analyzer analyze --deep --verbose src/
```

### Python API使用

```python
from tools.smart_code_analyzer import SmartCodeAnalyzer
from tools.smart_code_analyzer.core.config import SmartAnalysisConfig

# 创建分析器
analyzer = SmartCodeAnalyzer()

# 配置分析选项
config = SmartAnalysisConfig()
config.enable_deep_analysis = True
config.include_test_files = False
config.quality_threshold = 80.0

# 执行分析
results = analyzer.analyze_project('src/', config)

# 生成质量报告
report = analyzer.generate_quality_report(results)
print(f"项目质量评分: {report.overall_score:.1f}/100")

# 获取重构建议
refactoring_plan = analyzer.generate_refactoring_plan(results)
for suggestion in refactoring_plan.top_suggestions[:5]:
    print(f"- {suggestion.description} (优先级: {suggestion.priority})")
```

## 📊 分析指标详解

### 核心质量指标

| 指标 | 描述 | 理想范围 | 计算方法 |
|------|------|----------|----------|
| **整体质量评分** | 综合质量评估 | 85-100 | 加权多指标综合 |
| **可维护性指数** | 代码维护难度 | >65 | MI = 171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(L) + 50*sin(sqrt(2.4*CM)) |
| **圈复杂度** | 控制流复杂度 | 1-10 | McCabe复杂度计算 |
| **重复代码率** | 代码重复程度 | <5% | 基于AST的相似度分析 |
| **测试覆盖率** | 测试完整性 | >80% | 集成测试工具数据 |

### 代码异味检测

- **过长方法**: 方法行数 > 50
- **过长类**: 类行数 > 300
- **过深嵌套**: 嵌套深度 > 3层
- **重复代码块**: 相似度 > 80%
- **魔数使用**: 硬编码数字常量
- **未使用变量**: 定义但未使用的变量
- **过度耦合**: 模块间依赖过多

## 🎯 重构建议类型

### 方法级别重构
- **提取方法**: 将长方法拆分为多个小方法
- **内联方法**: 将不必要的方法调用内联
- **重命名方法**: 使用更有意义的方法名
- **参数简化**: 减少方法参数数量

### 类级别重构
- **提取类**: 从大类中提取内聚的功能
- **移动方法**: 将方法移动到更合适的类中
- **提取接口**: 为类提取接口定义
- **移除中间人**: 简化不必要的委托

### 模块级别重构
- **拆分模块**: 将大模块拆分为多个小模块
- **移动类**: 将类移动到更合适的模块
- **提取公共模块**: 创建共享的功能模块
- **重构依赖关系**: 优化模块间的依赖

## ⚙️ 配置选项

### 分析配置

```python
config = SmartAnalysisConfig()

# 基本设置
config.include_test_files = False      # 是否包含测试文件
config.max_file_size = 1000           # 最大文件大小(行)
config.parallel_processing = True     # 启用并行处理
config.cache_results = True          # 缓存分析结果

# 质量阈值
config.quality_threshold = 80.0       # 质量评分阈值
config.complexity_threshold = 10      # 复杂度阈值
config.duplication_threshold = 0.8    # 重复度阈值

# 重构建议
config.max_suggestions = 50           # 最大建议数量
config.min_confidence = 0.7           # 最小置信度
config.risk_tolerance = 'medium'      # 风险承受度
```

### 报告配置

```python
# 输出格式
config.report_format = 'html'         # html/json/xml
config.include_charts = True          # 包含图表
config.include_trends = True          # 包含趋势分析

# 详细程度
config.verbose_output = True          # 详细输出
config.include_raw_data = False       # 包含原始数据
config.max_detail_level = 3           # 最大详情级别
```

## 📈 报告和输出

### HTML质量报告
包含：
- 质量评分仪表板
- 代码异味统计图表
- 重构建议优先级列表
- 趋势分析图表
- 详细问题清单

### JSON结构化数据
```json
{
  "analysis_summary": {
    "total_files": 150,
    "average_quality_score": 87.5,
    "quality_distribution": {
      "excellent": 45,
      "good": 68,
      "fair": 25,
      "poor": 12
    }
  },
  "refactoring_plan": {
    "total_suggestions": 127,
    "high_priority": 23,
    "estimated_effort": "high",
    "top_suggestions": [...]
  }
}
```

## 🔗 与现有工具集成

### 与quality_check集成

```python
from tools.quality_check import QualityChecker
from tools.smart_code_analyzer import SmartCodeAnalyzer

# 组合使用两个工具
quality_checker = QualityChecker()
smart_analyzer = SmartCodeAnalyzer()

# 执行质量检查
quality_results = quality_checker.check_project('src/')

# 执行智能分析
smart_results = smart_analyzer.analyze_project('src/')

# 合并结果
combined_report = smart_analyzer.merge_with_quality_check(
    smart_results, quality_results
)
```

### 与smart_duplicate_detector集成

```python
from tools.smart_duplicate_detector import detect_clones
from tools.smart_code_analyzer import SmartCodeAnalyzer

# 获取重复检测结果
clone_results = detect_clones('src/')

# 整合到智能分析中
analyzer = SmartCodeAnalyzer()
integrated_results = analyzer.integrate_duplicate_analysis(
    clone_results
)
```

## 🎛️ 高级功能

### 增量分析
```python
# 只分析变更的文件
changed_files = get_git_changed_files()
incremental_results = analyzer.analyze_incremental(changed_files)
```

### 趋势分析
```python
# 分析代码质量趋势
trends = analyzer.analyze_quality_trends('src/', days=30)
print(f"质量趋势: {trends.direction}")  # improving/declining/stable
```

### 预测性分析
```python
# 预测未来的质量问题
predictions = analyzer.predict_quality_issues('src/', months_ahead=6)
for prediction in predictions.high_risk:
    print(f"预测问题: {prediction.description} (概率: {prediction.probability:.1%})")
```

## 📋 使用场景

### 1. 代码审查辅助
```bash
# 在Pull Request前进行质量检查
python -m tools.smart_code_analyzer analyze --pr-check --baseline main src/changed_files/
```

### 2. 重构规划
```bash
# 生成重构路线图
python -m tools.smart_code_analyzer refactor --plan --output refactor_plan.md src/
```

### 3. 持续监控
```bash
# CI/CD中集成质量门禁
python -m tools.smart_code_analyzer analyze --ci-gate --threshold 85.0 src/
```

### 4. 技术债务管理
```bash
# 识别和跟踪技术债务
python -m tools.smart_code_analyzer debt --track --baseline previous_release src/
```

## 🚀 扩展开发

### 添加新的分析器

```python
# 继承BaseAnalyzer
class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, file_path: str) -> AnalysisResult:
        # 实现自定义分析逻辑
        pass

    def get_metric_name(self) -> str:
        return "custom_metric"
```

### 添加新的重构器

```python
# 继承BaseRefactorer
class CustomRefactorer(BaseRefactorer):
    def can_apply(self, code_issue) -> bool:
        # 判断是否可以应用此重构
        pass

    def generate_suggestion(self, code_issue) -> RefactoringSuggestion:
        # 生成重构建议
        pass
```

## 📊 性能和扩展性

- **并行处理**: 支持多线程并行分析
- **增量分析**: 只分析变更的文件
- **结果缓存**: 缓存分析结果避免重复计算
- **内存优化**: 分批处理大项目
- **可扩展架构**: 插件化设计便于扩展

## 🔒 安全和隐私

- **本地分析**: 所有分析都在本地进行
- **无外部依赖**: 不发送代码到外部服务
- **结果隔离**: 分析结果仅本地存储
- **配置安全**: 支持敏感信息过滤

---

## 🎉 让代码分析更智能！

智能代码分析器整合了先进的代码分析技术和AI辅助功能，为RQA2025项目提供全方位的代码质量保障和智能化重构指导。

**立即开始使用**: `python -m tools.smart_code_analyzer analyze .`

---

*专为RQA2025项目设计的高级代码分析工具*
