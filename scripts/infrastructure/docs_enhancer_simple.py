#!/usr/bin/env python3
"""
简化版文档完善脚本
分析当前文档状态，补充API文档和使用指南
"""

import sys
import json
import ast
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DocsEnhancer:
    """文档完善器"""

    def __init__(self):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / 'docs'
        self.src_dir = self.project_root / 'src'
        self.report_dir = self.project_root / 'reports' / 'infrastructure'

        # 确保报告目录存在
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'doc_coverage': {},
            'api_docs': {},
            'usage_guides': {},
            'recommendations': [],
            'generated_files': []
        }

    def analyze_documentation_coverage(self) -> Dict[str, Any]:
        """分析文档覆盖率"""
        print("分析文档覆盖率...")

        coverage = {
            'total_modules': 0,
            'documented_modules': 0,
            'missing_docs': [],
            'outdated_docs': [],
            'api_docs': {},
            'usage_guides': {}
        }

        # 分析src/infrastructure目录
        infrastructure_dir = self.src_dir / 'infrastructure'
        if infrastructure_dir.exists():
            for module_path in infrastructure_dir.rglob('*.py'):
                if module_path.name.startswith('__'):
                    continue

                relative_path = module_path.relative_to(self.project_root)
                module_name = str(relative_path).replace('\\', '/').replace('.py', '')

                coverage['total_modules'] += 1

                # 检查是否有对应的文档
                doc_path = self.docs_dir / 'architecture' / \
                    'infrastructure' / f"{module_path.stem}.md"
                if doc_path.exists():
                    coverage['documented_modules'] += 1
                    # 检查文档是否过时
                    if self._is_doc_outdated(module_path, doc_path):
                        coverage['outdated_docs'].append(str(relative_path))
                else:
                    coverage['missing_docs'].append(str(relative_path))

                # 分析API文档
                api_docs = self._extract_api_docs(module_path)
                if api_docs:
                    coverage['api_docs'][module_name] = api_docs

        return coverage

    def _is_doc_outdated(self, module_path: Path, doc_path: Path) -> bool:
        """检查文档是否过时"""
        try:
            module_mtime = module_path.stat().st_mtime
            doc_mtime = doc_path.stat().st_mtime
            return module_mtime > doc_mtime
        except:
            return True

    def _extract_api_docs(self, module_path: Path) -> Dict[str, Any]:
        """提取API文档"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析Python代码
            tree = ast.parse(content)

            api_docs = {
                'classes': [],
                'functions': [],
                'constants': []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ""
                    api_docs['classes'].append({
                        'name': node.name,
                        'docstring': class_doc,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef):
                    func_doc = ast.get_docstring(node) or ""
                    api_docs['functions'].append({
                        'name': node.name,
                        'docstring': func_doc
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            api_docs['constants'].append({
                                'name': target.id,
                                'value': ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                            })

            return api_docs
        except Exception as e:
            print(f"警告: 解析API文档失败: {module_path} - {e}")
            return {}

    def generate_api_documentation(self) -> Dict[str, Any]:
        """生成API文档"""
        print("生成API文档...")

        api_docs = {}

        # 分析核心模块
        core_modules = [
            'src/infrastructure/logging',
            'src/infrastructure/config',
            'src/infrastructure/monitoring',
            'src/infrastructure/security',
            'src/infrastructure/database',
            'src/infrastructure/cache',
            'src/infrastructure/utils'
        ]

        for module_path in core_modules:
            full_path = self.project_root / module_path
            if full_path.exists():
                module_docs = self._generate_module_api_docs(full_path)
                if module_docs:
                    api_docs[module_path] = module_docs

        return api_docs

    def _generate_module_api_docs(self, module_path: Path) -> Dict[str, Any]:
        """为单个模块生成API文档"""
        module_docs = {
            'module_name': module_path.name,
            'description': '',
            'classes': [],
            'functions': [],
            'examples': []
        }

        # 查找模块的__init__.py或主要文件
        init_file = module_path / '__init__.py'
        if init_file.exists():
            module_docs.update(self._extract_api_docs(init_file))

        # 查找其他Python文件
        for py_file in module_path.glob('*.py'):
            if py_file.name != '__init__.py':
                file_docs = self._extract_api_docs(py_file)
                module_docs['classes'].extend(file_docs.get('classes', []))
                module_docs['functions'].extend(file_docs.get('functions', []))

        return module_docs

    def generate_usage_guides(self) -> Dict[str, Any]:
        """生成使用指南"""
        print("生成使用指南...")

        guides = {
            'quick_start': self._generate_quick_start_guide(),
            'configuration': self._generate_configuration_guide(),
            'logging': self._generate_logging_guide(),
            'monitoring': self._generate_monitoring_guide(),
            'security': self._generate_security_guide(),
            'troubleshooting': self._generate_troubleshooting_guide()
        }

        return guides

    def _generate_quick_start_guide(self) -> Dict[str, Any]:
        """生成快速开始指南"""
        return {
            'title': '基础设施层快速开始指南',
            'sections': [
                {
                    'title': '安装和配置',
                    'content': [
                        '1. 确保Python环境已安装',
                        '2. 安装项目依赖: pip install -r requirements.txt',
                        '3. 配置环境变量或配置文件',
                        '4. 运行测试验证安装: python scripts/testing/run_tests.py'
                    ]
                },
                {
                    'title': '基本使用',
                    'content': [
                        '1. 导入基础设施模块',
                        '2. 初始化配置管理器',
                        '3. 设置日志记录器',
                        '4. 配置监控服务',
                        '5. 启动健康检查'
                    ]
                },
                {
                    'title': '示例代码',
                    'code': '''
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import UnifiedLoggingInterface
from src.infrastructure.monitoring import MonitoringService

# 初始化配置
config = ConfigManager()
config.load_config('config/app.json')

# 设置日志
logging = UnifiedLoggingInterface()
logging.setup_logging(config.get('logging'))

# 启动监控
monitor = MonitoringService()
monitor.start_monitoring()
                    '''
                }
            ]
        }

    def _generate_configuration_guide(self) -> Dict[str, Any]:
        """生成配置指南"""
        return {
            'title': '配置管理使用指南',
            'sections': [
                {
                    'title': '配置文件格式',
                    'content': [
                        '支持JSON、YAML、INI格式',
                        '支持环境变量覆盖',
                        '支持配置验证和默认值'
                    ]
                },
                {
                    'title': '配置管理API',
                    'content': [
                        'ConfigManager.load_config() - 加载配置',
                        'ConfigManager.get() - 获取配置值',
                        'ConfigManager.set() - 设置配置值',
                        'ConfigManager.validate() - 验证配置'
                    ]
                }
            ]
        }

    def _generate_logging_guide(self) -> Dict[str, Any]:
        """生成日志指南"""
        return {
            'title': '日志系统使用指南',
            'sections': [
                {
                    'title': '日志级别',
                    'content': [
                        'DEBUG - 调试信息',
                        'INFO - 一般信息',
                        'WARNING - 警告信息',
                        'ERROR - 错误信息',
                        'CRITICAL - 严重错误'
                    ]
                },
                {
                    'title': '日志配置',
                    'content': [
                        '支持多种输出格式',
                        '支持日志轮转',
                        '支持结构化日志',
                        '支持日志聚合'
                    ]
                }
            ]
        }

    def _generate_monitoring_guide(self) -> Dict[str, Any]:
        """生成监控指南"""
        return {
            'title': '监控系统使用指南',
            'sections': [
                {
                    'title': '监控指标',
                    'content': [
                        '系统资源监控',
                        '应用性能监控',
                        '业务指标监控',
                        '自定义指标'
                    ]
                },
                {
                    'title': '告警规则',
                    'content': [
                        '阈值告警',
                        '趋势告警',
                        '异常检测',
                        '告警通知'
                    ]
                }
            ]
        }

    def _generate_security_guide(self) -> Dict[str, Any]:
        """生成安全指南"""
        return {
            'title': '安全模块使用指南',
            'sections': [
                {
                    'title': '认证授权',
                    'content': [
                        '用户认证',
                        '角色权限管理',
                        '会话管理',
                        '访问控制'
                    ]
                },
                {
                    'title': '数据安全',
                    'content': [
                        '数据加密',
                        '密钥管理',
                        '安全传输',
                        '数据脱敏'
                    ]
                }
            ]
        }

    def _generate_troubleshooting_guide(self) -> Dict[str, Any]:
        """生成故障排除指南"""
        return {
            'title': '故障排除指南',
            'sections': [
                {
                    'title': '常见问题',
                    'content': [
                        '配置加载失败',
                        '日志记录异常',
                        '监控数据丢失',
                        '性能问题'
                    ]
                },
                {
                    'title': '诊断工具',
                    'content': [
                        '健康检查',
                        '性能分析',
                        '日志分析',
                        '监控面板'
                    ]
                }
            ]
        }

    def generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        print("生成改进建议...")

        recommendations = [
            "补充缺失的模块文档",
            "更新过时的API文档",
            "添加更多使用示例",
            "完善故障排除指南",
            "增加性能调优指南",
            "补充安全最佳实践",
            "添加集成测试文档",
            "完善部署指南"
        ]

        return recommendations

    def save_documentation(self):
        """保存生成的文档"""
        print("保存文档...")

        # 保存API文档
        api_docs_file = self.report_dir / 'api_documentation.json'
        with open(api_docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results['api_docs'], f, ensure_ascii=False, indent=2)

        # 保存使用指南
        usage_guides_file = self.report_dir / 'usage_guides.json'
        with open(usage_guides_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results['usage_guides'], f, ensure_ascii=False, indent=2)

        # 保存完整报告
        report_file = self.report_dir / 'documentation_enhancement_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)

        self.analysis_results['generated_files'].extend([
            str(api_docs_file),
            str(usage_guides_file),
            str(report_file)
        ])

        print(f"文档已保存到: {self.report_dir}")

    def run(self):
        """运行文档完善流程"""
        print("开始文档完善流程...")

        try:
            # 分析文档覆盖率
            self.analysis_results['doc_coverage'] = self.analyze_documentation_coverage()

            # 生成API文档
            self.analysis_results['api_docs'] = self.generate_api_documentation()

            # 生成使用指南
            self.analysis_results['usage_guides'] = self.generate_usage_guides()

            # 生成建议
            self.analysis_results['recommendations'] = self.generate_recommendations()

            # 保存文档
            self.save_documentation()

            print("文档完善流程完成")

            # 输出摘要
            coverage = self.analysis_results['doc_coverage']
            print(f"\n=== 文档完善报告 ===")
            print(f"总模块数: {coverage['total_modules']}")
            print(f"已文档化模块: {coverage['documented_modules']}")
            if coverage['total_modules'] > 0:
                print(f"文档覆盖率: {coverage['documented_modules']/coverage['total_modules']*100:.1f}%")
            print(f"缺失文档: {len(coverage['missing_docs'])}")
            print(f"过时文档: {len(coverage['outdated_docs'])}")
            print(f"生成API文档: {len(self.analysis_results['api_docs'])}")
            print(f"生成使用指南: {len(self.analysis_results['usage_guides'])}")
            print(f"改进建议: {len(self.analysis_results['recommendations'])}")

        except Exception as e:
            print(f"文档完善流程失败: {e}")
            raise


if __name__ == '__main__':
    enhancer = DocsEnhancer()
    enhancer.run()
