# 自动化重构工具

一个智能的代码重构执行引擎，能够根据分析结果自动执行安全的代码重构操作，支持智能代码生成、批量重构和质量保障。

## 🎯 核心特性

### 🤖 智能重构执行
- **自动执行**: 根据智能代码分析器建议自动执行重构
- **安全重构**: 内置备份、验证、回滚机制
- **批量处理**: 支持批量重构多个文件和项目
- **增量执行**: 支持增量重构，避免重复处理

### 🛠️ 代码生成引擎
- **模板驱动**: 基于模板的智能代码生成
- **上下文感知**: 根据代码上下文生成合适的重构代码
- **模式识别**: 识别常见代码模式并生成优化代码
- **自定义扩展**: 支持自定义代码生成模板

### 🔒 安全保障机制
- **自动备份**: 重构前自动备份原代码
- **语法验证**: 重构后自动验证代码语法正确性
- **功能测试**: 可选的功能测试验证
- **一键回滚**: 出现问题时一键回滚到备份状态

### 📊 重构质量监控
- **执行统计**: 详细的重构执行统计和报告
- **质量验证**: 重构后的质量指标验证
- **影响分析**: 分析重构对系统的影响
- **性能监控**: 监控重构过程的性能表现

## 📁 目录结构

```
tools/auto_refactor/
├── core/                    # 核心引擎
│   ├── __init__.py
│   ├── refactor_engine.py   # 重构引擎核心
│   ├── safety_manager.py    # 安全管理器
│   └── config.py           # 配置管理
├── generators/              # 代码生成器
│   ├── __init__.py
│   ├── base_generator.py    # 基础生成器
│   ├── method_generator.py  # 方法生成器
│   ├── class_generator.py   # 类生成器
│   └── test_generator.py    # 测试生成器
├── executors/               # 重构执行器
│   ├── __init__.py
│   ├── base_executor.py     # 基础执行器
│   ├── extract_method.py    # 提取方法执行器
│   ├── split_class.py       # 拆分类执行器
│   └── rename_refactor.py   # 重命名执行器
├── validators/              # 验证器
│   ├── __init__.py
│   ├── syntax_validator.py  # 语法验证器
│   ├── semantic_validator.py # 语义验证器
│   └── test_validator.py    # 测试验证器
├── templates/               # 代码模板
│   ├── method_template.py   # 方法模板
│   ├── class_template.py    # 类模板
│   ├── test_template.py     # 测试模板
│   └── refactor_templates/  # 重构专用模板
├── __init__.py             # 包初始化
├── __main__.py             # 命令行入口
└── README.md               # 文档
```

## 🚀 快速开始

### 安装和环境要求

```bash
# 确保在项目根目录
cd /path/to/rqa2025

# 工具已集成到项目tools中，无需额外安装
python -m tools.auto_refactor --help
```

### 基本使用

```bash
# 分析并执行重构
python -m tools.auto_refactor analyze-and-refactor src/

# 只执行特定类型重构
python -m tools.auto_refactor refactor --type extract_method src/

# 安全模式执行（带备份和验证）
python -m tools.auto_refactor safe-refactor --backup --validate src/

# 生成代码而不执行重构
python -m tools.auto_refactor generate --type method --output generated_code.py
```

### Python API使用

```python
from tools.auto_refactor import AutoRefactorEngine
from tools.smart_code_analyzer import SmartCodeAnalyzer

# 创建重构引擎
engine = AutoRefactorEngine()

# 分析代码
analyzer = SmartCodeAnalyzer()
results = analyzer.analyze_project('src/')

# 执行自动重构
refactor_results = engine.execute_auto_refactor(results, safety_level='high')

# 查看重构报告
print(f"成功重构: {refactor_results.successful_refactors}")
print(f"跳过重构: {refactor_results.skipped_refactors}")
print(f"失败重构: {refactor_results.failed_refactors}")
```

## 📊 重构类型支持

### 方法级别重构
- **extract_method**: 提取长方法为多个小方法
- **inline_method**: 内联不必要的方法调用
- **rename_method**: 重命名方法为更有意义的名字
- **add_parameter**: 为方法添加参数
- **remove_parameter**: 移除方法参数

### 类级别重构
- **extract_class**: 从大类中提取内聚的功能
- **move_method**: 将方法移动到更合适的类
- **extract_interface**: 为类提取接口定义
- **collapse_hierarchy**: 简化不必要的继承层次
- **replace_inheritance_with_delegation**: 用委托替换继承

### 模块级别重构
- **move_class**: 将类移动到合适的模块
- **extract_module**: 创建新的功能模块
- **split_module**: 拆分过大的模块
- **merge_modules**: 合并功能相似的模块
- **rename_module**: 重命名模块

## ⚙️ 安全配置

### 安全级别设置

```python
from tools.auto_refactor.core.config import RefactorConfig

config = RefactorConfig()

# 高安全模式（推荐）
config.safety_level = 'high'
config.backup_enabled = True
config.validation_enabled = True
config.rollback_on_failure = True

# 中等安全模式
config.safety_level = 'medium'
config.backup_enabled = True
config.validation_enabled = False

# 低安全模式（仅用于可信环境）
config.safety_level = 'low'
config.backup_enabled = False
```

### 验证选项

```python
# 语法验证
config.syntax_validation = True

# 导入验证
config.import_validation = True

# 类型检查（如果有类型注解）
config.type_checking = True

# 测试执行
config.test_execution = True
config.test_timeout = 30  # 秒
```

## 🎯 重构执行流程

### 标准执行流程
1. **分析阶段**: 使用智能代码分析器分析代码质量
2. **建议生成**: 生成具体的重构建议和优先级排序
3. **安全检查**: 验证重构的安全性和可行性
4. **备份创建**: 为所有涉及文件创建备份
5. **重构执行**: 按优先级顺序执行重构操作
6. **验证阶段**: 验证重构结果的正确性
7. **报告生成**: 生成详细的重构执行报告

### 安全执行流程
```
分析代码 → 生成建议 → 风险评估 → 创建备份 → 执行重构 → 语法验证 → 语义验证 → 测试验证 → 生成报告
    ↓           ↓           ↓           ↓           ↓           ↓           ↓           ↓           ↓
   失败         失败        高风险      失败        失败        失败        失败        失败        完成
   退出         过滤        跳过        退出        回滚        回滚        回滚        回滚        ✅
```

## 📈 质量监控

### 执行指标监控

```python
# 获取重构执行统计
stats = engine.get_execution_stats()

print(f"总重构操作: {stats.total_operations}")
print(f"成功率: {stats.success_rate:.1%}")
print(f"平均执行时间: {stats.avg_execution_time:.2f}秒")
print(f"备份文件大小: {stats.backup_size_mb:.1f}MB")
```

### 质量变化跟踪

```python
# 重构前后的质量对比
quality_change = engine.compare_quality(before_results, after_results)

print(f"质量评分变化: {quality_change.score_change:+.1f}")
print(f"复杂度降低: {quality_change.complexity_reduction:.1f}%")
print(f"重复代码减少: {quality_change.duplication_reduction:.1f}%")
```

## 🔧 扩展开发

### 添加新的重构类型

```python
# 继承BaseRefactorExecutor
class CustomRefactorExecutor(BaseRefactorExecutor):
    def get_refactor_type(self) -> str:
        return "custom_refactor"

    def can_execute(self, suggestion) -> bool:
        # 检查是否可以执行此重构
        return suggestion.suggestion_type == "custom_refactor"

    def execute_safe(self, suggestion, context) -> RefactorResult:
        # 执行重构逻辑
        # 返回重构结果
        pass
```

### 添加代码生成器

```python
# 继承BaseCodeGenerator
class CustomCodeGenerator(BaseCodeGenerator):
    def generate_method(self, method_spec) -> str:
        # 生成方法代码
        template = self.get_template('custom_method')
        return template.render(**method_spec)

    def generate_class(self, class_spec) -> str:
        # 生成类代码
        template = self.get_template('custom_class')
        return template.render(**class_spec)
```

### 自定义验证器

```python
# 继承BaseValidator
class CustomValidator(BaseValidator):
    def validate_syntax(self, code: str) -> ValidationResult:
        # 自定义语法验证
        pass

    def validate_semantics(self, code: str, context) -> ValidationResult:
        # 自定义语义验证
        pass
```

## 📋 使用场景

### 1. 代码审查自动化
```bash
# 在CI/CD中自动执行安全重构
python -m tools.auto_refactor ci-refactor --safe --report ci_report.json
```

### 2. 批量代码优化
```bash
# 对整个项目执行批量重构
python -m tools.auto_refactor batch-refactor --project src/ --types "extract_method,split_class"
```

### 3. 新功能代码生成
```bash
# 生成CRUD操作代码
python -m tools.auto_refactor generate-crud --model User --output user_crud.py
```

### 4. 重构实验和验证
```bash
# 在隔离环境中测试重构效果
python -m tools.auto_refactor experiment --isolate --baseline master --branch refactor_branch
```

## 🎛️ 高级功能

### 并行执行
```bash
# 多线程并行执行重构
python -m tools.auto_refactor refactor --parallel 4 --batch-size 10 src/
```

### 增量重构
```bash
# 只重构变更的文件
python -m tools.auto_refactor incremental --since HEAD~1 src/
```

### 交互式重构
```bash
# 交互式选择和确认重构
python -m tools.auto_refactor interactive src/
```

### 重构流水线
```bash
# 定义重构流水线
python -m tools.auto_refactor pipeline --config refactor_pipeline.yaml
```

## 📊 性能和扩展性

- **并行处理**: 支持多线程并行执行重构操作
- **内存优化**: 增量处理大项目，避免内存溢出
- **缓存机制**: 缓存模板和分析结果，提高执行效率
- **可扩展架构**: 插件化设计，支持自定义扩展

## 🔒 安全和可靠性

- **自动备份**: 所有重构操作前自动创建备份
- **事务性执行**: 支持事务性重构，失败时自动回滚
- **验证机制**: 多层次验证确保重构结果正确性
- **权限控制**: 支持只读模式和权限检查

---

## 🚀 让重构更智能、更安全！

自动化重构工具结合了AI分析能力和安全执行机制，为RQA2025项目提供全自动化的代码质量提升解决方案。

**立即开始使用**: `python -m tools.auto_refactor analyze-and-refactor .`

---

*智能、安全、高效的自动化重构工具*
