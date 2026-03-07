# RQA API模块使用示例

## 📋 目录

- [快速开始](#快速开始)
- [OpenAPI文档生成](#openapi文档生成)
- [流程图生成](#流程图生成)
- [参数增强](#参数增强)
- [文档搜索](#文档搜索)
- [测试生成](#测试生成)
- [性能优化](#性能优化)
- [配置管理](#配置管理)

## 🚀 快速开始

### 安装依赖

```bash
# 确保在项目根目录
cd /path/to/rqa-project

# 激活conda环境
conda activate rqa

# 安装依赖（如果需要）
pip install -r requirements.txt
```

### 基本使用示例

```python
#!/usr/bin/env python3
"""
RQA API模块基础使用示例
"""

from pathlib import Path
from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGeneratorRefactored
from src.infrastructure.api.parameter_objects import DocumentationExportConfig

def main():
    """主函数示例"""

    print("🚀 RQA API模块使用示例")
    print("=" * 50)

    # 1. 创建生成器
    generator = RQAApiDocumentationGeneratorRefactored()
    print("✅ 创建OpenAPI文档生成器")

    # 2. 配置输出
    config = DocumentationExportConfig(
        output_dir="examples/output/docs",
        include_examples=True,
        include_statistics=True,
        format_types=["json", "yaml"]
    )
    print("✅ 配置文档导出参数")

    # 3. 生成文档
    try:
        files = generator.generate_documentation(config)
        print("✅ 文档生成完成")
        print(f"📁 生成的文件: {files}")

        # 4. 验证生成的文件
        for file_type, file_path in files.items():
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                print(f"  • {file_type.upper()}: {file_path} ({size} bytes)")
            else:
                print(f"  ❌ {file_type.upper()}: 文件未生成")

    except Exception as e:
        print(f"❌ 文档生成失败: {e}")
        return False

    print("\n🎉 示例执行完成！")
    return True

if __name__ == "__main__":
    main()
```

## 📄 OpenAPI文档生成

### 基本文档生成

```python
from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGeneratorRefactored

def generate_basic_docs():
    """生成基本的API文档"""

    generator = RQAApiDocumentationGeneratorRefactored()

    # 使用默认配置
    files = generator.generate_documentation()

    print("生成的文件:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")

    return files
```

### 高级配置生成

```python
from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGeneratorRefactored
from src.infrastructure.api.parameter_objects import DocumentationExportConfig

def generate_advanced_docs():
    """生成高级配置的API文档"""

    generator = RQAApiDocumentationGeneratorRefactored()

    # 自定义配置
    config = DocumentationExportConfig(
        output_dir="docs/advanced",
        include_examples=True,
        include_statistics=True,
        format_types=["json", "yaml"],
        pretty_print=True,
        include_metadata=True,
        compress=False,
        theme="professional"
    )

    files = generator.generate_documentation(config)

    return files
```

### 增量文档生成

```python
from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGeneratorRefactored

def generate_incremental_docs():
    """增量生成文档，避免重复生成"""

    generator = RQAApiDocumentationGeneratorRefactored()

    # 检查现有文件
    import os
    from pathlib import Path

    output_dir = Path("docs/api")
    if output_dir.exists():
        print("检测到现有文档目录")

        # 可以选择增量更新或完全重新生成
        choice = input("选择操作: [u]pdate 或 [r]egenerate? ")

        if choice.lower() == 'u':
            # 增量更新逻辑
            print("执行增量更新...")
        else:
            # 完全重新生成
            print("完全重新生成...")
            files = generator.generate_documentation()
    else:
        # 首次生成
        files = generator.generate_documentation()

    return files
```

## 🔄 流程图生成

### 数据服务流程图

```python
from src.infrastructure.api.flow_generation.flow_generators import DataServiceFlowGenerator

def generate_data_service_flow():
    """生成数据服务流程图"""

    generator = DataServiceFlowGenerator()

    # 生成流程图
    diagram = generator.generate()

    print(f"生成的流程图包含 {len(diagram.nodes)} 个节点, {len(diagram.edges)} 条边")

    # 导出为多种格式
    mermaid_files = generator.export_to_mermaid("docs/flows")
    json_files = generator.export_to_json("docs/flows")

    return {
        'diagram': diagram,
        'mermaid_files': mermaid_files,
        'json_files': json_files
    }
```

### 自定义流程图

```python
from src.infrastructure.api.flow_generation.node_builder import APIFlowNodeBuilder
from src.infrastructure.api.flow_generation.models import APIFlowDiagram

def create_custom_flow():
    """创建自定义流程图"""

    # 创建节点构建器
    builder = APIFlowNodeBuilder()

    # 添加节点
    builder.create_start_node(
        label="接收请求",
        position={"x": 100, "y": 100}
    )

    builder.create_process_node(
        node_id="validate",
        label="数据验证",
        position={"x": 250, "y": 100}
    )

    builder.create_decision_node(
        node_id="is_valid",
        label="数据是否有效?",
        position={"x": 400, "y": 100}
    )

    builder.create_process_node(
        node_id="process",
        label="处理数据",
        position={"x": 550, "y": 50}
    )

    builder.create_process_node(
        node_id="error",
        label="返回错误",
        position={"x": 550, "y": 150}
    )

    builder.create_end_node(
        label="完成",
        position={"x": 700, "y": 100}
    )

    # 添加连接
    builder.create_edge("start", "validate")
    builder.create_edge("validate", "is_valid")
    builder.create_edge("is_valid", "process", condition="是")
    builder.create_edge("is_valid", "error", condition="否")
    builder.create_edge("process", "end")
    builder.create_edge("error", "end")

    # 创建流程图对象
    diagram = APIFlowDiagram(
        id="custom_flow",
        title="自定义业务流程",
        description="演示自定义流程图创建",
        nodes=builder.nodes,
        edges=builder.edges
    )

    return diagram
```

## 🔧 参数增强

### 自动参数增强

```python
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import (
    ParameterEnhancer,
    APIParameterDocumentation
)

def enhance_parameters_example():
    """参数增强示例"""

    enhancer = ParameterEnhancer()

    # 创建各种类型的参数
    params = [
        APIParameterDocumentation("user_id", "string", True, "用户ID"),
        APIParameterDocumentation("email", "string", True, "邮箱地址", example="user@example.com"),
        APIParameterDocumentation("age", "integer", False, "年龄"),
        APIParameterDocumentation("price", "number", False, "价格"),
        APIParameterDocumentation("is_active", "boolean", False, "是否激活"),
        APIParameterDocumentation("tags", "array", False, "标签列表"),
        APIParameterDocumentation("config", "object", False, "配置对象"),
    ]

    enhanced_params = []

    for param in params:
        print(f"\n增强参数: {param.name} ({param.type})")

        # 增强前
        print(f"  原始示例值: {param.example}")

        # 执行增强
        enhancer.enhance_parameter(param)

        # 增强后
        print(f"  增强示例值: {param.example}")
        print(f"  生成约束: {param.constraints}")
        print(f"  验证规则: {param.validation_rules}")

        enhanced_params.append(param)

    # 显示缓存统计
    cache_stats = enhancer.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  缓存大小: {cache_stats['cache_size']}")
    print(f"  命中率: {cache_stats['hit_rate']:.2%}")

    return enhanced_params
```

### 批量参数处理

```python
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import ParameterEnhancer

def batch_parameter_processing():
    """批量参数处理示例"""

    enhancer = ParameterEnhancer()

    # 模拟API端点的参数列表
    api_endpoints = {
        "/users": {
            "user_id": ("string", "用户ID"),
            "username": ("string", "用户名"),
            "email": ("string", "邮箱"),
            "age": ("integer", "年龄"),
            "is_active": ("boolean", "是否激活"),
        },
        "/orders": {
            "order_id": ("string", "订单ID"),
            "user_id": ("string", "用户ID"),
            "amount": ("number", "订单金额"),
            "status": ("string", "订单状态"),
            "created_at": ("string", "创建时间"),
        }
    }

    all_enhanced_params = {}

    for endpoint, param_defs in api_endpoints.items():
        print(f"\n处理端点: {endpoint}")

        enhanced_params = []
        for param_name, (param_type, description) in param_defs.items():
            param = APIParameterDocumentation(
                param_name, param_type, True, description
            )

            enhancer.enhance_parameter(param)
            enhanced_params.append(param)

            print(f"  ✓ {param_name}: {param.example}")

        all_enhanced_params[endpoint] = enhanced_params

    return all_enhanced_params
```

## 🔍 文档搜索

### 基本搜索功能

```python
from src.infrastructure.api.documentation_search.search_engine import APIDocumentationSearchEngine
from src.infrastructure.api.parameter_objects import SearchConfig

def basic_search_example():
    """基本搜索功能示例"""

    engine = APIDocumentationSearchEngine()

    # 准备文档数据（示例）
    documents = {
        "user_api": {
            "path": "/api/v1/users",
            "method": "GET",
            "summary": "获取用户列表",
            "description": "获取系统中的所有用户列表，支持分页和筛选"
        },
        "auth_api": {
            "path": "/api/v1/auth/login",
            "method": "POST",
            "summary": "用户登录",
            "description": "验证用户凭据并返回访问令牌"
        }
    }

    # 执行搜索
    config = SearchConfig(
        query="用户登录",
        max_results=10,
        min_relevance_score=0.1
    )

    results = engine.search(documents, **config.__dict__)

    print(f"搜索 '{config.query}' 找到 {len(results)} 个结果:")

    for i, result in enumerate(results, 1):
        print(f"{i}. 相关度: {result.relevance_score:.3f}")
        print(f"   文档ID: {result.document_id}")
        print(f"   摘要: {result.snippet}")
        print()

    return results
```

### 高级搜索配置

```python
from src.infrastructure.api.parameter_objects import SearchConfig

def advanced_search_example():
    """高级搜索配置示例"""

    # 精确搜索
    precise_config = SearchConfig(
        query="POST /api/v1/auth/login",
        search_in_paths=True,
        search_in_methods=True,
        search_in_descriptions=False,
        case_sensitive=True,
        max_results=5
    )

    # 模糊搜索
    fuzzy_config = SearchConfig(
        query="authentication token",
        search_in_paths=False,
        search_in_methods=False,
        search_in_descriptions=True,
        search_in_parameters=True,
        search_in_responses=True,
        case_sensitive=False,
        max_results=20,
        min_relevance_score=0.3
    )

    return precise_config, fuzzy_config
```

## 🧪 测试生成

### 数据服务测试生成

```python
from src.infrastructure.api.test_generation.generators import DataServiceTestGenerator

def generate_data_service_tests():
    """生成数据服务测试套件"""

    generator = DataServiceTestGenerator()
    test_suite = generator.create_test_suite()

    print(f"生成的测试套件: {test_suite.name}")
    print(f"包含 {len(test_suite.scenarios)} 个测试场景")

    for scenario in test_suite.scenarios:
        print(f"  • {scenario.name}: {len(scenario.test_cases)} 个测试用例")

    # 导出测试
    files = generator.export_test_cases(
        format_type="json",
        output_dir="examples/output/tests"
    )

    print(f"测试导出到: {files}")
    return test_suite
```

### 自定义测试用例

```python
from src.infrastructure.api.test_generation.builders.base_builder import BaseTestBuilder
from src.infrastructure.api.parameter_objects import TestCaseConfig

def create_custom_test_case():
    """创建自定义测试用例"""

    builder = BaseTestBuilder()

    # 创建测试用例配置
    config = TestCaseConfig(
        title="用户注册完整流程测试",
        description="测试从用户注册到激活的完整流程",
        priority="high",
        category="integration",
        preconditions=[
            "数据库连接正常",
            "邮件服务可用",
            "用户注册API可用"
        ],
        test_steps=[
            {
                "step": 1,
                "action": "发送用户注册请求",
                "expected": "返回201状态码"
            },
            {
                "step": 2,
                "action": "验证用户数据已保存到数据库",
                "expected": "数据库中存在新用户记录"
            },
            {
                "step": 3,
                "action": "检查激活邮件是否发送",
                "expected": "邮件服务记录显示激活邮件已发送"
            },
            {
                "step": 4,
                "action": "使用激活链接激活账户",
                "expected": "账户状态变为已激活"
            }
        ],
        expected_results=[
            "用户成功注册",
            "收到激活邮件",
            "账户成功激活",
            "可以正常登录"
        ],
        tags=["registration", "email", "activation", "integration"]
    )

    # 创建测试用例
    test_case = builder.create_test_case(config)

    print("创建的测试用例:")
    print(f"  标题: {test_case.title}")
    print(f"  优先级: {test_case.priority}")
    print(f"  类别: {test_case.category}")
    print(f"  标签: {', '.join(test_case.tags)}")

    return test_case
```

## ⚡ 性能优化

### 缓存性能监控

```python
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import ParameterEnhancer
import time

def performance_monitoring_example():
    """性能监控示例"""

    enhancer = ParameterEnhancer()

    # 创建测试参数
    test_params = [
        APIParameterDocumentation(f"param_{i}", "string", False, f"测试参数{i}")
        for i in range(100)
    ]

    print("性能测试开始...")

    # 测试无缓存性能
    start_time = time.time()
    for _ in range(10):  # 10轮
        for param in test_params:
            enhancer._generate_example_value(param)
    no_cache_time = time.time() - start_time

    print(f"无缓存耗时: {no_cache_time:.3f}秒")

    # 显示缓存统计
    stats = enhancer.get_cache_stats()
    print(f"缓存命中率: {stats['hit_rate']:.2%}")
    print(f"缓存大小: {stats['cache_size']}")

    # 测试有缓存性能
    start_time = time.time()
    for _ in range(10):  # 再来10轮
        for param in test_params:
            enhancer._generate_example_value(param)
    with_cache_time = time.time() - start_time

    print(f"有缓存耗时: {with_cache_time:.3f}秒")
    print(f"性能提升: {no_cache_time / with_cache_time:.1f}倍")

    return stats
```

### 缓存管理

```python
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import ParameterEnhancer

def cache_management_example():
    """缓存管理示例"""

    enhancer = ParameterEnhancer()

    # 执行一些操作
    for i in range(50):
        param = APIParameterDocumentation(f"test_{i}", "string", False, f"测试{i}")
        enhancer.enhance_parameter(param)

    print("缓存状态:")
    stats = enhancer.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 清理缓存
    print("\n清理缓存...")
    enhancer.clear_cache()

    print("清理后缓存状态:")
    stats = enhancer.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

## ⚙️ 配置管理

### 参数对象最佳实践

```python
from src.infrastructure.api.parameter_objects import (
    DocumentationExportConfig,
    SearchConfig,
    TestCaseConfig,
    FlowDiagramConfig
)

def configuration_best_practices():
    """配置管理最佳实践"""

    # 1. 文档导出配置
    doc_config = DocumentationExportConfig(
        output_dir="docs/production",
        include_examples=True,
        include_statistics=True,
        format_types=["json", "yaml"],
        pretty_print=True,
        theme="professional"
    )

    # 2. 搜索配置
    search_config = SearchConfig(
        query="user management",
        max_results=25,
        min_relevance_score=0.4,
        case_sensitive=False
    )

    # 3. 流程图配置
    flow_config = FlowDiagramConfig(
        diagram_id="user_flow",
        title="用户管理流程",
        diagram_type="sequential",
        orientation="TB",
        include_legend=True,
        include_metadata=True
    )

    # 4. 测试用例配置
    test_config = TestCaseConfig(
        title="用户CRUD操作测试",
        description="测试用户创建、读取、更新、删除操作",
        priority="high",
        category="functional",
        preconditions=["数据库已初始化", "用户API可用"],
        tags=["user", "crud", "api"]
    )

    return {
        "doc_config": doc_config,
        "search_config": search_config,
        "flow_config": flow_config,
        "test_config": test_config
    }
```

### 错误处理和重试

```python
import time
from pathlib import Path

def robust_documentation_generation():
    """健壮的文档生成，包含错误处理和重试"""

    from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGeneratorRefactored
    from src.infrastructure.api.parameter_objects import DocumentationExportConfig

    generator = RQAApiDocumentationGeneratorRefactored()

    config = DocumentationExportConfig(
        output_dir="docs/robust",
        format_types=["json", "yaml"]
    )

    max_retries = 3
    retry_delay = 1  # 秒

    for attempt in range(max_retries):
        try:
            print(f"尝试生成文档 (第{attempt + 1}次)...")

            # 确保输出目录存在
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

            # 生成文档
            files = generator.generate_documentation(config)

            # 验证生成的文件
            all_exist = True
            for file_path in files.values():
                if not Path(file_path).exists():
                    all_exist = False
                    break

            if all_exist:
                print("✅ 文档生成成功")
                return files
            else:
                raise FileNotFoundError("部分文件未生成")

        except Exception as e:
            print(f"❌ 尝试 {attempt + 1} 失败: {e}")

            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("达到最大重试次数，放弃")
                raise

    return None
```

## 🔧 实用工具函数

### 文件操作辅助函数

```python
from pathlib import Path
from typing import Dict, Any
import json

def save_with_backup(data: Dict[str, Any], file_path: str, backup: bool = True):
    """安全保存文件，支持备份"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 创建备份
    if backup and path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.backup")
        backup_path.write_text(path.read_text(), encoding='utf-8')

    # 保存新文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ 文件已保存: {file_path}")

def load_with_fallback(file_path: str, fallback_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """安全加载文件，支持降级"""

    path = Path(file_path)

    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载文件失败 {file_path}: {e}")

    # 返回降级数据或空字典
    return fallback_data or {}

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个配置字典"""

    result = {}

    for config in configs:
        for key, value in config.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[key] = merge_configs(result[key], value)
            else:
                # 直接覆盖
                result[key] = value

    return result
```

---

## 📞 获取帮助

如果您在使用过程中遇到问题，可以：

1. 查看详细的API文档：`docs/api/README.md`
2. 运行测试用例验证功能：`pytest tests/unit/infrastructure/api/`
3. 查看代码质量报告：`python scripts/code_quality_monitor.py`
4. 查阅错误日志和调试信息

## 🎯 最佳实践总结

1. **始终使用参数对象**替代长参数列表
2. **合理配置缓存**以提升性能
3. **添加错误处理**提高程序健壮性
4. **定期监控质量**预防问题积累
5. **编写测试用例**确保功能正确性

---

**示例版本**: 2.1.0
**更新日期**: 2025-10-27
**示例作者**: AI Assistant
