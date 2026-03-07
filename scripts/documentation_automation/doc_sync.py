#!/usr/bin/env python3
"""
RQA2025 文档自动化同步工具

提供文档与代码的自动化同步功能，包括：
- 自动提取代码中的接口和类定义
- 更新文档中的代码示例
- 同步API文档
- 生成变更报告
"""

import os
import re
import json
import logging
import ast
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocSync:
    """文档同步器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.docs_dir = self.project_root / "docs" / "architecture"
        self.src_dir = self.project_root / "src"
        self.templates_dir = self.project_root / "scripts" / "documentation_automation" / "templates"

        # 创建模板目录
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def sync_all_docs(self) -> Dict[str, Any]:
        """同步所有文档"""
        logger.info("开始同步所有文档...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'sync_results': {},
            'summary': {},
            'changes': []
        }

        # 同步各层架构文档
        layers = ['core', 'data', 'features', 'ml', 'trading', 'risk']
        for layer in layers:
            try:
                doc_file = self.docs_dir / f"{layer}_layer_architecture_design.md"
                if doc_file.exists() or layer == 'core':
                    if layer == 'core':
                        doc_file = self.docs_dir / "core_layer_architecture_design.md"

                    result = self.sync_layer_doc(layer, doc_file)
                    results['sync_results'][layer] = result

                    if result.get('changes'):
                        results['changes'].extend(result['changes'])

            except Exception as e:
                logger.error(f"同步 {layer} 层文档失败: {e}")
                results['sync_results'][layer] = {
                    'status': 'error',
                    'error': str(e)
                }

        # 生成摘要
        results['summary'] = self._generate_sync_summary(results)

        # 保存同步报告
        self._save_sync_report(results)

        logger.info("文档同步完成")
        return results

    def sync_layer_doc(self, layer: str, doc_file: Path) -> Dict[str, Any]:
        """同步指定层的文档"""
        result = {
            'status': 'success',
            'changes': [],
            'updated_sections': []
        }

        if not doc_file.exists():
            logger.warning(f"文档文件不存在: {doc_file}")
            return result

        logger.info(f"同步 {layer} 层文档: {doc_file.name}")

        # 读取原始文档
        original_content = doc_file.read_text(encoding='utf-8')

        # 同步不同类型的文档内容
        updated_content = original_content

        # 1. 同步接口定义
        updated_content, interface_changes = self._sync_interfaces(updated_content, layer)
        result['changes'].extend(interface_changes)

        # 2. 同步代码示例
        updated_content, code_changes = self._sync_code_examples(updated_content, layer)
        result['changes'].extend(code_changes)

        # 3. 同步API文档
        updated_content, api_changes = self._sync_api_docs(updated_content, layer)
        result['changes'].extend(api_changes)

        # 4. 更新版本信息
        updated_content, version_changes = self._sync_version_info(updated_content, layer)
        result['changes'].extend(version_changes)

        # 保存更新后的文档
        if updated_content != original_content:
            doc_file.write_text(updated_content, encoding='utf-8')
            result['updated_sections'] = ['interfaces', 'code_examples', 'api_docs', 'version_info']
            logger.info(f"文档已更新: {doc_file.name}")
        else:
            logger.info(f"文档无需更新: {doc_file.name}")

        return result

    def _sync_interfaces(self, content: str, layer: str) -> Tuple[str, List[str]]:
        """同步接口定义"""
        changes = []

        # 查找接口定义部分
        interface_section_pattern = r'(## \d+\..*?接口.*?\n)(.*?)(?=\n## \d+\.|$)'
        match = re.search(interface_section_pattern, content, re.DOTALL)

        if match:
            section_header = match.group(1)
            section_content = match.group(2)

            # 提取当前文档中的接口
            doc_interfaces = self._extract_interfaces_from_doc(section_content)

            # 提取代码中的接口
            code_interfaces = self._extract_interfaces_from_code(layer)

            # 比较差异
            missing_in_doc = code_interfaces - doc_interfaces
            outdated_in_doc = doc_interfaces - code_interfaces

            if missing_in_doc or outdated_in_doc:
                # 生成更新的接口文档
                updated_section = self._generate_interface_section(layer, code_interfaces)
                content = content.replace(match.group(0), section_header + updated_section)

                changes.append(
                    f"接口同步: 添加 {len(missing_in_doc)} 个新接口, 移除 {len(outdated_in_doc)} 个过时接口")

        return content, changes

    def _sync_code_examples(self, content: str, layer: str) -> Tuple[str, List[str]]:
        """同步代码示例"""
        changes = []

        # 查找代码示例部分
        code_example_pattern = r'(```python\s*\n)(.*?)(\n```)'
        matches = re.findall(code_example_pattern, content, re.DOTALL)

        for i, (start, code_block, end) in enumerate(matches):
            # 验证代码示例语法
            try:
                ast.parse(code_block)
            except SyntaxError as e:
                # 尝试修复代码示例
                fixed_code = self._fix_code_example(code_block, layer)
                if fixed_code != code_block:
                    old_block = f"{start}{code_block}{end}"
                    new_block = f"{start}{fixed_code}{end}"
                    content = content.replace(old_block, new_block)
                    changes.append(f"修复代码示例语法错误 (示例 {i+1})")

            # 验证导入语句
            import_lines = [line for line in code_block.split('\n')
                            if line.strip().startswith(('from ', 'import '))]

            for import_line in import_lines:
                if not self._validate_import(import_line, layer):
                    # 尝试修复导入
                    fixed_import = self._fix_import_statement(import_line, layer)
                    if fixed_import != import_line:
                        content = content.replace(import_line, fixed_import)
                        changes.append(f"修复导入语句: {import_line.strip()}")

        return content, changes

    def _sync_api_docs(self, content: str, layer: str) -> Tuple[str, List[str]]:
        """同步API文档"""
        changes = []

        # 查找API文档部分
        api_section_pattern = r'(## \d+\..*?API.*?\n)(.*?)(?=\n## \d+\.|$)'
        match = re.search(api_section_pattern, content, re.DOTALL)

        if match:
            section_header = match.group(1)
            section_content = match.group(2)

            # 提取代码中的API
            apis = self._extract_apis_from_code(layer)

            # 生成API文档
            updated_api_docs = self._generate_api_docs(layer, apis)
            content = content.replace(match.group(0), section_header + updated_api_docs)

            changes.append(f"API文档同步: 更新 {len(apis)} 个API接口")

        return content, changes

    def _sync_version_info(self, content: str, layer: str) -> Tuple[str, List[str]]:
        """同步版本信息"""
        changes = []

        # 查找版本信息
        version_pattern = r'(\*\*文档版本\*\*: )([^\n]+)'
        match = re.search(version_pattern, content)

        if match:
            current_version = match.group(2)
            # 从代码中提取版本信息
            code_version = self._extract_version_from_code(layer)

            if code_version and code_version != current_version:
                content = content.replace(match.group(0), match.group(1) + code_version)
                changes.append(f"版本信息更新: {current_version} -> {code_version}")

        # 更新最后修改时间
        timestamp_pattern = r'(\*\*最后更新\*\*: )([^\n]+)'
        match = re.search(timestamp_pattern, content)

        if match:
            current_time = datetime.now().strftime("%Y年%m月%d日")
            content = content.replace(match.group(0), match.group(1) + current_time)
            changes.append("更新文档修改时间戳")

        return content, changes

    def _extract_interfaces_from_doc(self, content: str) -> Set[str]:
        """从文档中提取接口名称"""
        interfaces = set()

        # 查找接口类定义
        patterns = [
            r'class (I\w+)\(IBusinessAdapter\)',
            r'class (I\w+)\(ICoreComponent\)',
            r'class (I\w+)\(ABC\)',
            r'class (I\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            interfaces.update(matches)

        return interfaces

    def _extract_interfaces_from_code(self, layer: str) -> Set[str]:
        """从代码中提取接口定义"""
        interfaces = set()

        # 扫描指定层的代码文件
        layer_dir = self.src_dir / layer
        if not layer_dir.exists():
            return interfaces

        for py_file in layer_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')

                # 查找接口类定义
                class_pattern = r'class (I\w+).*?:'
                matches = re.findall(class_pattern, content)
                interfaces.update(matches)

            except Exception as e:
                logger.warning(f"读取文件失败 {py_file}: {e}")

        return interfaces

    def _extract_apis_from_code(self, layer: str) -> List[Dict[str, Any]]:
        """从代码中提取API信息"""
        apis = []

        layer_dir = self.src_dir / layer
        if not layer_dir.exists():
            return apis

        for py_file in layer_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')

                # 查找函数定义
                func_pattern = r'def (\w+)\([^)]*\):(.*?)(?=\n\s*def|\n\s*class|\n\s*@|\n\s*$)'
                matches = re.findall(func_pattern, content, re.DOTALL)

                for func_name, func_body in matches:
                    # 提取函数签名和文档字符串
                    api_info = {
                        'name': func_name,
                        'file': str(py_file.relative_to(self.project_root)),
                        'signature': f"def {func_name}(...)",
                        'docstring': self._extract_docstring(func_body)
                    }
                    apis.append(api_info)

            except Exception as e:
                logger.warning(f"提取API失败 {py_file}: {e}")

        return apis

    def _extract_version_from_code(self, layer: str) -> Optional[str]:
        """从代码中提取版本信息"""
        layer_dir = self.src_dir / layer
        init_file = layer_dir / "__init__.py"

        if init_file.exists():
            try:
                content = init_file.read_text(encoding='utf-8')
                version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'
                match = re.search(version_pattern, content)
                if match:
                    return match.group(1)
            except Exception as e:
                logger.warning(f"提取版本失败 {init_file}: {e}")

        return None

    def _generate_interface_section(self, layer: str, interfaces: Set[str]) -> str:
        """生成接口文档部分"""
        section = "\n### 接口定义\n\n"

        if interfaces:
            section += "本层定义的主要接口如下：\n\n"
            for interface in sorted(interfaces):
                section += f"- **`{interface}`**: {self._get_interface_description(interface)}\n"
            section += "\n"
        else:
            section += "暂无接口定义。\n\n"

        return section

    def _generate_api_docs(self, layer: str, apis: List[Dict[str, Any]]) -> str:
        """生成API文档"""
        section = "\n### API接口\n\n"

        if apis:
            section += "| API名称 | 文件位置 | 描述 |\n"
            section += "|---------|----------|------|\n"

            for api in apis[:20]:  # 限制显示数量
                desc = api.get('docstring', '').split('\n')[
                    0][:50] if api.get('docstring') else '暂无描述'
                section += f"| `{api['name']}` | {api['file']} | {desc} |\n"

            section += "\n"
        else:
            section += "暂无API接口。\n\n"

        return section

    def _get_interface_description(self, interface_name: str) -> str:
        """获取接口描述"""
        descriptions = {
            'IBusinessAdapter': '业务层适配器统一接口',
            'ICoreComponent': '核心组件统一接口',
            'IModelsProvider': '模型层服务提供接口',
            'IFeaturesProvider': '特征层服务提供接口',
            'IDataProvider': '数据层服务提供接口'
        }
        return descriptions.get(interface_name, '自定义接口')

    def _extract_docstring(self, function_body: str) -> str:
        """提取函数文档字符串"""
        lines = function_body.strip().split('\n')
        docstring_lines = []

        in_docstring = False
        quote_type = None

        for line in lines:
            stripped = line.strip()

            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    quote_type = '"""' if stripped.startswith('"""') else "'''"
                    docstring_lines.append(stripped)
                elif stripped.startswith('"') or stripped.startswith("'"):
                    # 单行docstring
                    return stripped.strip('"\'')
            else:
                docstring_lines.append(line)
                if quote_type in stripped:
                    break

        if docstring_lines:
            return '\n'.join(docstring_lines).strip(quote_type).strip()

        return ""

    def _fix_code_example(self, code_block: str, layer: str) -> str:
        """修复代码示例"""
        # 简单的语法修复
        try:
            # 尝试解析
            ast.parse(code_block)
            return code_block
        except SyntaxError as e:
            logger.warning(f"代码示例语法错误: {e}")
            # 这里可以实现更复杂的修复逻辑
            return code_block

    def _validate_import(self, import_line: str, layer: str) -> bool:
        """验证导入语句"""
        # 简单的验证逻辑
        if 'from src.' in import_line or 'import src.' in import_line:
            # 检查模块是否存在
            try:
                # 解析导入路径
                if 'from ' in import_line:
                    parts = import_line.replace('from ', '').split(' import ')[0].split('.')
                else:
                    parts = import_line.replace('import ', '').split('.')[1:]

                module_path = self.src_dir
                for part in parts[1:]:  # 跳过 'src'
                    module_path = module_path / part

                if module_path.exists() or (module_path.parent / f"{module_path.name}.py").exists():
                    return True
            except:
                pass

        return False

    def _fix_import_statement(self, import_line: str, layer: str) -> str:
        """修复导入语句"""
        # 这里可以实现导入修复逻辑
        # 例如：修正错误的模块路径
        return import_line

    def _generate_sync_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成同步摘要"""
        total_layers = len(results.get('sync_results', {}))
        successful_syncs = sum(1 for r in results.get('sync_results', {}).values()
                               if r.get('status') == 'success')
        total_changes = len(results.get('changes', []))

        return {
            'total_layers': total_layers,
            'successful_syncs': successful_syncs,
            'total_changes': total_changes,
            'sync_rate': (successful_syncs / total_layers * 100) if total_layers > 0 else 0
        }

    def _save_sync_report(self, results: Dict[str, Any]):
        """保存同步报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "reports" / "technical" / \
            "doc_sync" / f"sync_report_{timestamp}.json"

        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"同步报告已保存: {report_file}")

    def generate_sync_template(self, layer: str) -> str:
        """生成同步模板"""
        template = """# 层架构设计文档

## 📋 文档概述

**文档版本**: v1.0.0
**更新时间**: {update_time}
**文档状态**: ✅ 已同步
**设计理念**: 业务流程驱动 + 统一基础设施集成
**核心创新**: 层专用适配器 + 标准化接口设计
**架构一致性**: ⭐⭐⭐⭐⭐ (100%与基础设施层保持一致)

---

## 1. 模块定位

### 1.1 业务定位
层是RQA2025量化交易系统的核心组件，负责核心功能。

### 1.2 技术定位
- **架构层次**: 业务服务层
- **核心职责**: 提供核心业务功能
- **技术栈**: Python + 相关技术栈
- **部署模式**: 微服务架构

---

## 2. 架构设计理念

### 2.1 统一基础设施集成原则
通过层适配器实现与统一基础设施层的深度集成。

### 2.2 标准化接口设计原则
遵循统一的层间接口规范，确保系统的一致性和可扩展性。

### 2.3 业务流程驱动原则
基于量化交易的核心业务流程设计所有功能模块。

---

## 3. 核心组件设计

### 3.1 层适配器统一基础设施集成

#### 功能特性
- **统一基础设施访问**: 提供层专用的基础设施服务访问接口
- **服务降级保障**: 内置降级服务，确保系统高可用性
- **集中化管理**: 基础设施集成逻辑集中管理
- **标准化接口**: 统一的API接口，降低学习成本

#### 核心实现
```python
# src/core/integration/layer_adapter.py
class LayerAdapter(BaseBusinessAdapter):
    "层专用适配器"

    def __init__(self):
        super().__init__(BusinessLayerType.LAYER)
        self._init_layer_infrastructure()

    def get_layer_cache_manager(self):
        "获取层专用缓存管理器"
        return self.get_infrastructure_services().get('cache_manager')

    def get_layer_config_manager(self):
        "获取层专用配置管理器"
        return self._config_manager
```

---

## 4. 接口定义

### 主要接口
<!-- 此部分将在同步时自动更新 -->

---

## 5. API接口

### 核心API
<!-- 此部分将在同步时自动更新 -->

---

## 6. 使用示例

### 基本使用
```python
# 获取层适配器
adapter = get_layer_adapter()

# 使用核心功能
result = adapter.some_core_function()
```

---

## 7. 部署和运维

### 部署配置
<!-- 部署配置将在此处添加 -->

---

## 8. 版本信息

**文档版本**: v1.0.0
**最后更新**: {update_time}
"""

        return template.format(update_time=datetime.now().strftime("%Y年%m月%d日"))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 文档同步工具')
    parser.add_argument('--project-root', help='项目根目录路径')
    parser.add_argument('--layer', help='指定要同步的层')
    parser.add_argument('--generate-template', action='store_true', help='生成同步模板')

    args = parser.parse_args()

    # 创建同步器
    sync = DocSync(args.project_root)

    if args.generate_template and args.layer:
        # 生成同步模板
        template = sync.generate_sync_template(args.layer)
        template_file = sync.templates_dir / f"{args.layer}_template.md"
        template_file.write_text(template, encoding='utf-8')
        print(f"模板已生成: {template_file}")

    elif args.layer:
        # 同步指定层
        doc_file = sync.docs_dir / f"{args.layer}_layer_architecture_design.md"
        result = sync.sync_layer_doc(args.layer, doc_file)
        print(f"层同步完成: {args.layer}")
        print(f"变更: {len(result.get('changes', []))} 项")

    else:
        # 同步所有文档
        results = sync.sync_all_docs()
        summary = results.get('summary', {})
        print("文档同步完成")
        print(f"成功同步: {summary.get('successful_syncs', 0)}/{summary.get('total_layers', 0)}")
        print(f"总变更: {summary.get('total_changes', 0)} 项")


if __name__ == "__main__":
    main()
