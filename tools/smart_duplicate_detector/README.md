# 智能重复代码检测工具

一个高级的代码克隆检测、重构建议生成和自动修复工具，专为RQA2025项目设计。

## 特性

### 🔍 智能检测
- **AST结构分析**: 基于Python AST的精确代码结构分析
- **多维度相似度**: 文本、AST结构、语义相似度综合计算
- **变量标准化**: 智能识别和标准化变量名，提高检测准确性
- **多种克隆类型**: 精确克隆、相似克隆、语义克隆分类识别

### 📋 自动重构建议
- **方法提取**: 自动识别可提取为公共方法的重复代码
- **类提取**: 建议创建工具类封装重复逻辑
- **配置提取**: 识别配置相关重复并建议提取到配置文件
- **父类方法**: 建议将重复方法提取到父类

### 🛠️ 高级功能
- **并行处理**: 支持多线程并行分析，提高检测速度
- **可配置阈值**: 灵活配置相似度阈值和检测参数
- **多种输出格式**: 支持JSON、XML、HTML格式报告
- **增量分析**: 支持增量检测，避免重复分析

## 安装

工具已集成到RQA2025项目中，无需额外安装。

## 使用方法

### 命令行使用

```bash
# 基本检测
python -m tools.smart_duplicate_detector src/infrastructure/cache

# 指定输出格式
python -m tools.smart_duplicate_detector --format json --output report.json src/

# 使用不同预设配置
python -m tools.smart_duplicate_detector --preset strict src/

# 比较两个文件
python -m tools.smart_duplicate_detector --compare-files file1.py file2.py

# 详细输出
python -m tools.smart_duplicate_detector --verbose src/
```

### 预设配置

- `strict`: 严格模式，只检测高度相似的代码
- `normal`: 正常模式，平衡检测精度和性能
- `relaxed`: 宽松模式，检测更多潜在重复
- `performance`: 性能模式，快速检测

### Python API使用

```python
from tools.smart_duplicate_detector import detect_clones
from tools.smart_duplicate_detector.core.config import SmartDuplicateConfig

# 创建配置
config = SmartDuplicateConfig()
config.get_preset_config('normal')

# 检测克隆
result = detect_clones('src/infrastructure/cache', config)

# 输出结果
print(f"发现 {len(result.clone_groups)} 个克隆组")
print(f"涉及 {result.total_fragments_analyzed} 个代码片段")

# 获取重构建议
refactoring_ops = result.get_refactoring_opportunities()
for op in refactoring_ops[:5]:  # 显示前5个建议
    print(f"- {op['description']} (影响: {op['impact']})")
```

## 输出格式

### JSON格式
```json
{
  "detection_time": "2025-09-21T21:40:00",
  "total_files_analyzed": 10,
  "total_fragments_analyzed": 176,
  "total_clone_groups": 6,
  "statistics": {
    "total_groups": 6,
    "total_fragments": 45,
    "avg_similarity": 0.85,
    "clone_types": {"exact": 4, "similar": 2},
    "files_affected": 8
  },
  "clone_groups": [...],
  "refactoring_opportunities": [...]
}
```

### HTML格式
生成包含统计图表和详细信息的网页报告。

## 配置选项

### 相似度阈值
```python
config.similarity.exact_clone = 0.95      # 精确克隆阈值
config.similarity.similar_clone = 0.8     # 相似克隆阈值
config.similarity.semantic_clone = 0.6    # 语义克隆阈值
```

### 分析配置
```python
config.analysis.min_fragment_size = 5     # 最小的代码片段行数
config.analysis.extract_functions = True  # 是否提取函数
config.analysis.normalize_variables = True # 是否标准化变量名
```

### 性能配置
```python
config.performance.parallel_processing = True  # 启用并行处理
config.performance.max_workers = 4           # 最大工作线程数
config.performance.similarity_cache_size = 10000  # 相似度缓存大小
```

## 架构

```
tools/smart_duplicate_detector/
├── core/                    # 核心数据结构
│   ├── code_fragment.py     # 代码片段表示
│   ├── similarity_metrics.py # 相似度计算
│   ├── detection_result.py  # 检测结果
│   └── config.py           # 配置管理
├── analyzers/              # 分析器
│   ├── base_analyzer.py    # 基础分析器
│   ├── fragment_extractor.py # 片段提取器
│   ├── similarity_analyzer.py # 相似度分析器
│   └── clone_detector.py   # 克隆检测器
├── refactorers/            # 重构器
│   ├── base_refactorer.py  # 基础重构器
│   └── method_extractor.py # 方法提取器
├── utils/                  # 工具类
│   └── ast_utils.py        # AST工具
└── __main__.py            # 命令行入口
```

## 算法说明

### 相似度计算
1. **文本相似度**: 使用序列匹配算法计算文本相似性
2. **AST相似度**: 比较抽象语法树结构
3. **语义相似度**: 基于变量使用模式、控制结构等语义特征
4. **综合评分**: 加权组合多种相似度指标

### 克隆检测流程
1. **代码片段提取**: 从Python文件中提取函数、类、方法等代码片段
2. **标准化处理**: 移除注释、标准化缩进、变量名标准化
3. **相似度计算**: 计算所有片段间的相似度
4. **聚类分析**: 将相似片段归类为克隆组
5. **重构建议**: 基于克隆组特征生成重构建议

## 与原有工具比较

| 特性 | 原有工具 | 智能工具 |
|------|----------|----------|
| 相似度算法 | 基本哈希+序列匹配 | 多维度相似度计算 |
| AST分析 | 无 | 深度AST结构分析 |
| 语义理解 | 无 | 变量标准化和语义分析 |
| 重构建议 | 无 | 自动生成具体建议 |
| 输出格式 | 单一 | JSON/XML/HTML多种格式 |
| 并行处理 | 无 | 支持多线程并行 |
| 可配置性 | 有限 | 高度可配置 |

## 性能优化

- **增量缓存**: 缓存相似度计算结果，避免重复计算
- **并行处理**: 多线程并行分析，提高检测速度
- **智能过滤**: 预过滤明显不同的代码片段
- **内存优化**: 分批处理大文件，避免内存溢出

## 扩展开发

### 添加新的相似度算法
```python
# 在 similarity_metrics.py 中添加新方法
@staticmethod
def custom_similarity(fragment1, fragment2) -> float:
    # 实现自定义相似度算法
    pass
```

### 添加新的重构器
```python
# 继承 BaseRefactorer
class CustomRefactorer(BaseRefactorer):
    def can_refactor(self, clone_group) -> bool:
        # 判断是否可以应用此重构
        pass

    def generate_suggestion(self, clone_group):
        # 生成重构建议
        pass
```

## 许可证

专为RQA2025项目开发使用。

## 贡献

欢迎提交问题和改进建议！

---

**智能重复代码检测工具** - 让代码重构更智能、更高效！
