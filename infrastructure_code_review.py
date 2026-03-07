"""
基础设施层代码审查工具

根据架构设计文档对代码实现进行全面审查
检查代码组织、重叠冗余、接口一致性等
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import hashlib


class InfrastructureCodeReviewer:
    """基础设施层代码审查器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.issues = []
        self.metrics = {}
        self.code_analysis = {}

    def perform_comprehensive_review(self) -> Dict[str, Any]:
        """执行全面代码审查"""
        print('🔍 开始基础设施层代码审查')
        print('=' * 60)

        review_results = {
            'architecture_compliance': self._check_architecture_compliance(),
            'code_organization': self._analyze_code_organization(),
            'redundancy_analysis': self._analyze_code_redundancy(),
            'interface_consistency': self._check_interface_consistency(),
            'import_structure': self._analyze_import_structure(),
            'quality_metrics': self._calculate_quality_metrics(),
            'recommendations': []
        }

        # 生成综合建议
        review_results['recommendations'] = self._generate_recommendations(review_results)

        # 保存审查报告
        self._save_review_report(review_results)

        print(f'\\n✅ 代码审查完成，发现 {len(self.issues)} 个问题')
        return review_results

    def _check_architecture_compliance(self) -> Dict[str, Any]:
        """检查架构一致性"""
        print('🏗️ 检查架构一致性...')

        compliance = {
            'interface_inheritance': self._check_interface_inheritance(),
            'directory_structure': self._check_directory_structure(),
            'naming_conventions': self._check_naming_conventions(),
            'module_boundaries': self._check_module_boundaries()
        }

        return compliance

    def _check_interface_inheritance(self) -> Dict[str, Any]:
        """检查接口继承"""
        interface_patterns = {
            'ComponentFactory': 'BaseComponentFactory',
            'Factory': 'BaseFactory',
            'Manager': 'BaseManager',
            'Service': 'BaseService',
            'Handler': 'BaseHandler',
            'Provider': 'BaseProvider'
        }

        inheritance_issues = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查类定义
                        class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
                        classes = re.findall(class_pattern, content)

                        for class_name in classes:
                            # 检查是否应该继承特定接口
                            for pattern, base_class in interface_patterns.items():
                                if pattern in class_name and pattern != class_name:
                                    # 检查是否正确继承
                                    if f'class {class_name}(' in content:
                                        if base_class not in content:
                                            inheritance_issues.append({
                                                'file': rel_path,
                                                'class': class_name,
                                                'expected_base': base_class,
                                                'issue': 'missing_interface_inheritance'
                                            })

                    except Exception as e:
                        continue

        return {
            'status': 'compliant' if not inheritance_issues else 'issues_found',
            'issues': inheritance_issues,
            'compliance_rate': 1.0 - len(inheritance_issues) / max(1, self._count_total_classes())
        }

    def _check_directory_structure(self) -> Dict[str, Any]:
        """检查目录结构"""
        expected_structure = {
            'interfaces': ['component_factory.py', 'factory_pattern.py', 'manager_pattern.py',
                           'service_pattern.py', 'handler_pattern.py', 'provider_pattern.py'],
            'cache': ['core/', 'managers/', 'services/', 'monitoring/', 'config/', 'utils/'],
            'config': ['core/', 'loaders/', 'mergers/', 'storage/', 'version/', 'monitoring/',
                       'services/', 'interfaces/', 'utils/', 'tests/', 'web/', 'tools/'],
            'logging': ['foundation/', 'data/', 'config/', 'security/', 'system/',
                        'distributed/', 'utils/'],
            'error': ['foundation/', 'recovery/', 'testing/', 'storage/', 'security/',
                      'components/', 'utils/'],
            'health': ['core/', 'services/'],
            'resource': ['core/', 'monitors/', 'services/']
        }

        structure_issues = []

        for module, expected_subdirs in expected_structure.items():
            module_path = self.infra_dir / module
            if module_path.exists():
                actual_items = []
                if module_path.is_dir():
                    actual_items = [item.name for item in module_path.iterdir() if item.is_dir()]
                    actual_items.extend([item.name for item in module_path.iterdir()
                                        if item.is_file() and item.name.endswith('.py')])

                missing_items = set(expected_subdirs) - set(actual_items)
                if missing_items:
                    structure_issues.append({
                        'module': module,
                        'missing': list(missing_items),
                        'issue': 'missing_expected_structure'
                    })

        return {
            'status': 'compliant' if not structure_issues else 'issues_found',
            'issues': structure_issues
        }

    def _check_naming_conventions(self) -> Dict[str, Any]:
        """检查命名规范"""
        naming_issues = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查类命名
                        class_pattern = r'class\s+([A-Z][a-zA-Z0-9_]*)(?:\([^)]*\))?:'
                        classes = re.findall(class_pattern, content)

                        for class_name in classes:
                            # 检查接口命名 (I前缀)
                            if class_name.startswith('I') and len(class_name) > 1:
                                continue  # 接口命名正确
                            elif not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                                naming_issues.append({
                                    'file': rel_path,
                                    'type': 'class',
                                    'name': class_name,
                                    'issue': 'invalid_class_name'
                                })

                        # 检查方法命名
                        method_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                        methods = re.findall(method_pattern, content)

                        for method_name in methods:
                            if not re.match(r'^[a-z_][a-zA-Z0-9_]*$', method_name):
                                naming_issues.append({
                                    'file': rel_path,
                                    'type': 'method',
                                    'name': method_name,
                                    'issue': 'invalid_method_name'
                                })

                    except Exception as e:
                        continue

        return {
            'status': 'compliant' if not naming_issues else 'issues_found',
            'issues': naming_issues
        }

    def _check_module_boundaries(self) -> Dict[str, Any]:
        """检查模块边界"""
        boundary_issues = []

        # 检查跨模块直接导入
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))
                        current_module = rel_path.split('/')[0] if '/' in rel_path else 'root'

                        # 检查from导入
                        from_imports = re.findall(r'from\s+src\.infrastructure\.(\w+)\.', content)
                        for imported_module in from_imports:
                            if imported_module != current_module:
                                # 检查是否合理的跨模块导入
                                allowed_cross_imports = {
                                    'interfaces': ['*'],  # interfaces可以被任何人导入
                                    'base': ['*'],        # base可以被任何人导入
                                }

                                if current_module not in allowed_cross_imports.get(imported_module, []):
                                    boundary_issues.append({
                                        'file': rel_path,
                                        'from_module': current_module,
                                        'to_module': imported_module,
                                        'issue': 'cross_module_import'
                                    })

                    except Exception as e:
                        continue

        return {
            'status': 'compliant' if not boundary_issues else 'issues_found',
            'issues': boundary_issues
        }

    def _analyze_code_organization(self) -> Dict[str, Any]:
        """分析代码组织"""
        print('📁 分析代码组织...')

        organization_metrics = {
            'file_sizes': self._analyze_file_sizes(),
            'function_lengths': self._analyze_function_lengths(),
            'class_complexity': self._analyze_class_complexity(),
            'module_cohesion': self._analyze_module_cohesion()
        }

        return organization_metrics

    def _analyze_file_sizes(self) -> Dict[str, Any]:
        """分析文件大小"""
        file_sizes = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        size = file_path.stat().st_size
                        rel_path = str(file_path.relative_to(self.infra_dir))

                        file_sizes.append({
                            'file': rel_path,
                            'size_bytes': size,
                            'size_kb': size / 1024,
                            'size_lines': self._count_lines(file_path)
                        })
                    except Exception:
                        continue

        # 找出异常大的文件
        large_files = [f for f in file_sizes if f['size_kb'] > 100]  # >100KB
        very_large_files = [f for f in file_sizes if f['size_kb'] > 500]  # >500KB

        return {
            'total_files': len(file_sizes),
            'avg_size_kb': sum(f['size_kb'] for f in file_sizes) / len(file_sizes) if file_sizes else 0,
            'large_files': large_files,
            'very_large_files': very_large_files,
            'size_distribution': self._categorize_file_sizes(file_sizes)
        }

    def _analyze_function_lengths(self) -> Dict[str, Any]:
        """分析函数长度"""
        function_lengths = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取函数定义和内容
                        functions = re.findall(
                            r'def\s+\w+.*?:.*?(?=\\n\\n|\\n\s*def|\\n\s*@|\\n\s*class|\\Z)', content, re.DOTALL)

                        for func in functions:
                            lines = len(func.split('\\n'))
                            function_lengths.append({
                                'file': rel_path,
                                'function': func.split('\\n')[0] if func else 'unknown',
                                'lines': lines
                            })

                    except Exception:
                        continue

        # 分析长函数
        long_functions = [f for f in function_lengths if f['lines'] > 50]
        very_long_functions = [f for f in function_lengths if f['lines'] > 100]

        return {
            'total_functions': len(function_lengths),
            'avg_length': sum(f['lines'] for f in function_lengths) / len(function_lengths) if function_lengths else 0,
            'long_functions': long_functions,
            'very_long_functions': very_long_functions
        }

    def _analyze_class_complexity(self) -> Dict[str, Any]:
        """分析类复杂度"""
        class_complexity = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取类定义
                        classes = re.findall(
                            r'class\s+\w+.*?:.*?(?=\\n\\n|\\n\s*class|\\Z)', content, re.DOTALL)

                        for cls in classes:
                            lines = len(cls.split('\\n'))
                            methods = len(re.findall(r'\\n\s*def\s+', cls))

                            class_complexity.append({
                                'file': rel_path,
                                'class': cls.split('\\n')[0] if cls else 'unknown',
                                'lines': lines,
                                'methods': methods,
                                'complexity_score': lines * 0.3 + methods * 2  # 简单的复杂度评分
                            })

                    except Exception:
                        continue

        # 分析复杂类
        complex_classes = [c for c in class_complexity if c['complexity_score'] > 50]
        very_complex_classes = [c for c in class_complexity if c['complexity_score'] > 100]

        return {
            'total_classes': len(class_complexity),
            'avg_complexity': sum(c['complexity_score'] for c in class_complexity) / len(class_complexity) if class_complexity else 0,
            'complex_classes': complex_classes,
            'very_complex_classes': very_complex_classes
        }

    def _analyze_module_cohesion(self) -> Dict[str, Any]:
        """分析模块内聚性"""
        module_metrics = defaultdict(
            lambda: {'files': 0, 'classes': 0, 'functions': 0, 'imports': 0})

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))
                        module = rel_path.split('/')[0] if '/' in rel_path else 'root'

                        # 统计各种元素
                        classes = len(re.findall(r'^class\s+', content, re.MULTILINE))
                        functions = len(re.findall(r'^def\s+', content, re.MULTILINE))
                        imports = len(re.findall(r'^(from|import)\s+', content, re.MULTILINE))

                        module_metrics[module]['files'] += 1
                        module_metrics[module]['classes'] += classes
                        module_metrics[module]['functions'] += functions
                        module_metrics[module]['imports'] += imports

                    except Exception:
                        continue

        # 计算内聚性指标
        cohesion_scores = {}
        for module, metrics in module_metrics.items():
            # 内聚性评分：类和函数密度
            total_elements = metrics['classes'] + metrics['functions']
            cohesion_scores[module] = {
                'files': metrics['files'],
                'total_elements': total_elements,
                'elements_per_file': total_elements / metrics['files'] if metrics['files'] > 0 else 0,
                'import_density': metrics['imports'] / metrics['files'] if metrics['files'] > 0 else 0
            }

        return dict(cohesion_scores)

    def _analyze_code_redundancy(self) -> Dict[str, Any]:
        """分析代码冗余"""
        print('🔄 分析代码冗余...')

        redundancy = {
            'duplicate_functions': self._find_duplicate_functions(),
            'duplicate_classes': self._find_duplicate_classes(),
            'similar_code_blocks': self._find_similar_code_blocks(),
            'redundant_imports': self._find_redundant_imports()
        }

        return redundancy

    def _find_duplicate_functions(self) -> List[Dict[str, Any]]:
        """查找重复函数"""
        function_signatures = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取函数签名和实现
                        functions = re.findall(
                            r'(def\s+\w+.*?:.*?(?=\\n\\n|\\n\s*def|\\n\s*@|\\n\s*class|\\Z))', content, re.DOTALL)

                        for func in functions:
                            # 创建函数签名哈希
                            func_hash = hashlib.md5(func.encode()).hexdigest()[:16]
                            function_signatures[func_hash].append({
                                'file': rel_path,
                                'signature': func.split('\\n')[0] if func else 'unknown',
                                'content_hash': func_hash
                            })

                    except Exception:
                        continue

        # 找出重复的函数
        duplicates = []
        for func_hash, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                duplicates.append({
                    'function_hash': func_hash,
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })

        return duplicates

    def _find_duplicate_classes(self) -> List[Dict[str, Any]]:
        """查找重复类"""
        class_signatures = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取类定义
                        classes = re.findall(
                            r'(class\s+\w+.*?:.*?(?=\\n\\n|\\n\s*class|\\Z))', content, re.DOTALL)

                        for cls in classes:
                            # 创建类签名哈希
                            cls_hash = hashlib.md5(cls.encode()).hexdigest()[:16]
                            class_signatures[cls_hash].append({
                                'file': rel_path,
                                'class_name': cls.split('\\n')[0] if cls else 'unknown',
                                'content_hash': cls_hash
                            })

                    except Exception:
                        continue

        # 找出重复的类
        duplicates = []
        for cls_hash, occurrences in class_signatures.items():
            if len(occurrences) > 1:
                duplicates.append({
                    'class_hash': cls_hash,
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })

        return duplicates

    def _find_similar_code_blocks(self) -> List[Dict[str, Any]]:
        """查找相似的代码块"""
        # 这里使用简单的相似性检测
        code_blocks = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取代码块 (连续的非空行)
                        lines = content.split('\\n')
                        current_block = []
                        in_block = False

                        for line in lines:
                            stripped = line.strip()
                            if stripped and not stripped.startswith('#'):
                                if not in_block:
                                    in_block = True
                                    current_block = [line]
                                else:
                                    current_block.append(line)
                            elif in_block and current_block:
                                # 结束当前块
                                if len(current_block) > 5:  # 只分析较长的块
                                    block_content = '\\n'.join(current_block)
                                    block_hash = hashlib.md5(
                                        block_content.encode()).hexdigest()[:16]

                                    code_blocks.append({
                                        'file': rel_path,
                                        'content': block_content[:200] + '...' if len(block_content) > 200 else block_content,
                                        'lines': len(current_block),
                                        'hash': block_hash
                                    })

                                current_block = []
                                in_block = False

                    except Exception:
                        continue

        # 统计相似块
        hash_counts = Counter(block['hash'] for block in code_blocks)
        similar_blocks = []

        for block_hash, count in hash_counts.items():
            if count > 1:
                similar_instances = [block for block in code_blocks if block['hash'] == block_hash]
                similar_blocks.append({
                    'pattern_hash': block_hash,
                    'occurrences': similar_instances,
                    'count': count,
                    'sample_content': similar_instances[0]['content']
                })

        return similar_blocks

    def _find_redundant_imports(self) -> List[Dict[str, Any]]:
        """查找冗余导入"""
        redundant_imports = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取所有导入
                        imports = re.findall(r'^(from|import)\s+(.+)', content, re.MULTILINE)
                        imported_items = {}

                        for imp_type, imp_content in imports:
                            if imp_type == 'from':
                                # from module import item1, item2
                                parts = imp_content.split(' import ')
                                if len(parts) == 2:
                                    module = parts[0].strip()
                                    items = [item.strip() for item in parts[1].split(',')]
                                    for item in items:
                                        key = f"{module}.{item}"
                                        if key in imported_items:
                                            redundant_imports.append({
                                                'file': rel_path,
                                                'redundant_import': key,
                                                'issue': 'duplicate_import'
                                            })
                                        else:
                                            imported_items[key] = True

                    except Exception:
                        continue

        return redundant_imports

    def _check_interface_consistency(self) -> Dict[str, Any]:
        """检查接口一致性"""
        print('🔗 检查接口一致性...')

        interface_consistency = {
            'interface_implementations': self._check_interface_implementations(),
            'method_signatures': self._check_method_signatures(),
            'abstract_method_coverage': self._check_abstract_method_coverage()
        }

        return interface_consistency

    def _check_interface_implementations(self) -> Dict[str, Any]:
        """检查接口实现"""
        interface_implementations = []

        # 定义已知的接口
        known_interfaces = {
            'IComponentFactory': 'BaseComponentFactory',
            'IFactory': 'BaseFactory',
            'IManager': 'BaseManager',
            'IService': 'BaseService',
            'IHandler': 'BaseHandler',
            'IProvider': 'BaseProvider'
        }

        for interface, base_class in known_interfaces.items():
            implementations = []

            for root, dirs, files in os.walk(self.infra_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                            rel_path = str(file_path.relative_to(self.infra_dir))

                            if base_class in content:
                                implementations.append(rel_path)

                        except Exception:
                            continue

            interface_implementations.append({
                'interface': interface,
                'base_class': base_class,
                'implementations': implementations,
                'implementation_count': len(implementations)
            })

        return {
            'interfaces': interface_implementations,
            'total_interfaces': len(known_interfaces),
            'total_implementations': sum(impl['implementation_count'] for impl in interface_implementations)
        }

    def _check_method_signatures(self) -> Dict[str, Any]:
        """检查方法签名一致性"""
        method_signatures = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 提取方法签名
                        methods = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)', content)

                        for method_name, params in methods:
                            signature = f"{method_name}({params})"
                            method_signatures[method_name].append({
                                'file': rel_path,
                                'signature': signature
                            })

                    except Exception:
                        continue

        # 找出不一致的方法签名
        inconsistent_methods = []

        for method_name, signatures in method_signatures.items():
            if len(signatures) > 1:
                # 检查签名是否一致
                unique_signatures = set(sig['signature'] for sig in signatures)
                if len(unique_signatures) > 1:
                    inconsistent_methods.append({
                        'method': method_name,
                        'signatures': signatures,
                        'unique_signatures': list(unique_signatures)
                    })

        return {
            'total_methods': len(method_signatures),
            'inconsistent_methods': inconsistent_methods,
            'consistency_rate': 1.0 - len(inconsistent_methods) / len(method_signatures) if method_signatures else 1.0
        }

    def _check_abstract_method_coverage(self) -> Dict[str, Any]:
        """检查抽象方法覆盖"""
        # 这里可以实现抽象方法的检查逻辑
        return {'status': 'not_implemented'}

    def _analyze_import_structure(self) -> Dict[str, Any]:
        """分析导入结构"""
        print('📦 分析导入结构...')

        import_analysis = {
            'import_patterns': self._analyze_import_patterns(),
            'circular_dependencies': self._detect_circular_dependencies(),
            'unused_imports': self._find_unused_imports()
        }

        return import_analysis

    def _analyze_import_patterns(self) -> Dict[str, Any]:
        """分析导入模式"""
        import_stats = defaultdict(int)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 统计导入类型
                        from_imports = len(re.findall(r'^from\s+', content, re.MULTILINE))
                        direct_imports = len(re.findall(r'^import\s+', content, re.MULTILINE))
                        wildcard_imports = len(re.findall(r'import\s+\*', content))

                        import_stats['from_imports'] += from_imports
                        import_stats['direct_imports'] += direct_imports
                        import_stats['wildcard_imports'] += wildcard_imports
                        import_stats['total_imports'] += from_imports + direct_imports

                    except Exception:
                        continue

        return dict(import_stats)

    def _detect_circular_dependencies(self) -> List[Dict[str, Any]]:
        """检测循环依赖"""
        # 简化的循环依赖检测
        circular_deps = []

        # 分析导入关系
        import_graph = defaultdict(set)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))
                        module_name = rel_path.replace('/', '.').replace('.py', '')

                        # 提取导入的模块
                        imports = re.findall(r'from\s+src\.infrastructure\.(\w+)', content)
                        for imported_module in imports:
                            import_graph[module_name].add(imported_module)

                    except Exception:
                        continue

        # 检查简单的循环依赖 (A->B->A)
        for module, deps in import_graph.items():
            for dep in deps:
                if dep in import_graph and module in import_graph[dep]:
                    circular_deps.append({
                        'cycle': [module, dep, module],
                        'description': f"{module} ↔ {dep}"
                    })

        return circular_deps

    def _find_unused_imports(self) -> List[Dict[str, Any]]:
        """查找未使用的导入"""
        # 这里可以实现更复杂的未使用导入检测
        return []

    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """计算质量指标"""
        print('📊 计算质量指标...')

        metrics = {
            'complexity_metrics': self._calculate_complexity_metrics(),
            'maintainability_index': self._calculate_maintainability_index(),
            'test_coverage_estimate': self._estimate_test_coverage(),
            'documentation_coverage': self._calculate_documentation_coverage()
        }

        return metrics

    def _calculate_complexity_metrics(self) -> Dict[str, Any]:
        """计算复杂度指标"""
        total_lines = 0
        total_files = 0
        total_functions = 0
        total_classes = 0

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    total_files += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = len(content.split('\\n'))
                        functions = len(re.findall(r'^def\s+', content, re.MULTILINE))
                        classes = len(re.findall(r'^class\s+', content, re.MULTILINE))

                        total_lines += lines
                        total_functions += functions
                        total_classes += classes

                    except Exception:
                        continue

        return {
            'total_lines': total_lines,
            'total_files': total_files,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'avg_lines_per_file': total_lines / total_files if total_files > 0 else 0,
            'avg_functions_per_file': total_functions / total_files if total_files > 0 else 0,
            'avg_classes_per_file': total_classes / total_files if total_files > 0 else 0
        }

    def _calculate_maintainability_index(self) -> float:
        """计算可维护性指数"""
        # 简化的可维护性计算
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * Cyclomatic Complexity - 16.2 * ln(Lines of Code)
        # 这里使用简化的版本
        complexity_metrics = self._calculate_complexity_metrics()

        lines_of_code = complexity_metrics['total_lines']
        num_functions = complexity_metrics['total_functions']

        # 简化的可维护性评分 (0-100)
        mi_score = 100 - (lines_of_code * 0.01) - (num_functions * 0.1)
        mi_score = max(0, min(100, mi_score))

        return mi_score

    def _estimate_test_coverage(self) -> float:
        """估算测试覆盖率"""
        # 检查测试文件
        test_files = 0
        total_files = 0

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    total_files += 1
                    if 'test' in file.lower():
                        test_files += 1

        # 估算测试覆盖率
        if total_files > 0:
            coverage = (test_files / total_files) * 100
            return min(100, coverage * 10)  # 放大估算
        return 0

    def _calculate_documentation_coverage(self) -> float:
        """计算文档覆盖率"""
        total_functions = 0
        documented_functions = 0

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        functions = re.findall(r'def\s+\w+', content)
                        total_functions += len(functions)

                        # 检查文档字符串
                        for func in functions:
                            func_pattern = f'{func}.*?:(.*?(?=\\n\\n|\\n\s*def|\\n\s*@|\\n\s*class|\\Z))'
                            func_match = re.search(func_pattern, content, re.DOTALL)
                            if func_match:
                                func_content = func_match.group(1)
                                if '"""' in func_content or "'''" in func_content:
                                    documented_functions += 1

                    except Exception:
                        continue

        if total_functions > 0:
            return (documented_functions / total_functions) * 100
        return 0

    def _generate_recommendations(self, review_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []

        # 基于架构一致性
        arch_compliance = review_results.get('architecture_compliance', {})
        if arch_compliance.get('interface_inheritance', {}).get('status') == 'issues_found':
            recommendations.append({
                'priority': 'high',
                'category': 'architecture',
                'title': '修复接口继承问题',
                'description': f'发现 {len(arch_compliance["interface_inheritance"]["issues"])} 个接口继承问题',
                'actions': ['为所有Factory类添加正确的基类继承', '统一接口实现模式']
            })

        # 基于代码组织
        code_org = review_results.get('code_organization', {})
        file_sizes = code_org.get('file_sizes', {})
        if file_sizes.get('very_large_files'):
            recommendations.append({
                'priority': 'medium',
                'category': 'organization',
                'title': '拆分大文件',
                'description': f'发现 {len(file_sizes["very_large_files"])} 个超大文件',
                'actions': ['将大文件拆分为多个小文件', '按功能划分模块']
            })

        # 基于冗余分析
        redundancy = review_results.get('redundancy_analysis', {})
        duplicate_funcs = redundancy.get('duplicate_functions', [])
        if duplicate_funcs:
            recommendations.append({
                'priority': 'high',
                'category': 'redundancy',
                'title': '消除重复函数',
                'description': f'发现 {len(duplicate_funcs)} 组重复函数',
                'actions': ['提取公共函数到工具模块', '建立统一的工具函数库']
            })

        # 基于质量指标
        quality = review_results.get('quality_metrics', {})
        maintainability = quality.get('maintainability_index', 100)
        if maintainability < 70:
            recommendations.append({
                'priority': 'medium',
                'category': 'quality',
                'title': '提升代码可维护性',
                'description': f'可维护性指数: {maintainability:.1f}/100',
                'actions': ['减少函数复杂度', '增加文档覆盖', '优化代码结构']
            })

        # 通用建议
        recommendations.extend([
            {
                'priority': 'medium',
                'category': 'testing',
                'title': '完善测试覆盖',
                'description': '当前测试覆盖率需要提升',
                'actions': ['建立单元测试套件', '实现集成测试', '设置端到端测试']
            },
            {
                'priority': 'low',
                'category': 'documentation',
                'title': '完善文档',
                'description': '提升代码文档覆盖率',
                'actions': ['为所有公共方法添加文档', '建立API文档', '编写使用指南']
            }
        ])

        return recommendations

    def _count_total_classes(self) -> int:
        """统计总类数"""
        total_classes = 0
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        total_classes += len(re.findall(r'^class\s+', content, re.MULTILINE))
                    except Exception:
                        continue
        return total_classes

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception:
            return 0

    def _categorize_file_sizes(self, file_sizes: List[Dict]) -> Dict[str, int]:
        """分类文件大小"""
        categories = {'small': 0, 'medium': 0, 'large': 0, 'very_large': 0}

        for file_info in file_sizes:
            kb = file_info['size_kb']
            if kb < 10:
                categories['small'] += 1
            elif kb < 50:
                categories['medium'] += 1
            elif kb < 200:
                categories['large'] += 1
            else:
                categories['very_large'] += 1

        return categories

    def _save_review_report(self, review_results: Dict[str, Any]):
        """保存审查报告"""
        report = {
            'review_timestamp': str(Path('.').stat().st_mtime),
            'summary': {
                'total_issues': len(self.issues),
                'architecture_compliance': review_results.get('architecture_compliance', {}).get('interface_inheritance', {}).get('compliance_rate', 0),
                'code_quality_score': 100 - (len(self.issues) * 2),  # 简化的质量评分
                'maintainability_index': review_results.get('quality_metrics', {}).get('maintainability_index', 0)
            },
            'detailed_results': review_results
        }

        with open('infrastructure_code_review_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print('✅ 代码审查报告已保存: infrastructure_code_review_report.json')


def main():
    """主函数"""
    reviewer = InfrastructureCodeReviewer()
    results = reviewer.perform_comprehensive_review()

    # 输出关键发现
    print('\\n📋 审查结果摘要:')

    # 架构一致性
    arch = results.get('architecture_compliance', {})
    inheritance = arch.get('interface_inheritance', {})
    print(f'🏗️ 架构一致性: {inheritance.get("compliance_rate", 0):.1%}')

    # 代码组织
    org = results.get('code_organization', {})
    file_sizes = org.get('file_sizes', {})
    print(f'📁 文件组织: {file_sizes.get("total_files", 0)} 个文件, 平均 {file_sizes.get("avg_size_kb", 0):.1f}KB')

    # 代码冗余
    redundancy = results.get('redundancy_analysis', {})
    dup_funcs = len(redundancy.get('duplicate_functions', []))
    dup_classes = len(redundancy.get('duplicate_classes', []))
    print(f'🔄 代码冗余: {dup_funcs} 组重复函数, {dup_classes} 组重复类')

    # 接口一致性
    interface = results.get('interface_consistency', {})
    impl = interface.get('interface_implementations', {})
    print(f'🔗 接口一致性: {impl.get("total_implementations", 0)} 个接口实现')

    # 质量指标
    quality = results.get('quality_metrics', {})
    mi = quality.get('maintainability_index', 0)
    doc_cov = quality.get('documentation_coverage', 0)
    print(f'📊 质量指标: 可维护性 {mi:.1f}/100, 文档覆盖 {doc_cov:.1f}%')

    # 改进建议
    recommendations = results.get('recommendations', [])
    print(f'💡 改进建议: {len(recommendations)} 项')

    print('\\n✅ 基础设施层代码审查完成！')


if __name__ == "__main__":
    main()
