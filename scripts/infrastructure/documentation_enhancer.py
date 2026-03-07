#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层文档完善脚本
自动生成API文档、使用指南、架构文档等
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructureDocumentationEnhancer:
    """基础设施层文档完善器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.docs_dir = self.project_root / "docs"
        self.documentation_log = self.project_root / "backup" / \
            "documentation_enhancement" / "documentation_log.json"

        # 创建备份目录
        self.documentation_log.parent.mkdir(parents=True, exist_ok=True)

        # 文档类型
        self.documentation_types = {
            'api_docs': self._generate_api_documentation,
            'usage_guides': self._generate_usage_guides,
            'architecture_docs': self._generate_architecture_documentation,
            'deployment_guides': self._generate_deployment_guides,
            'troubleshooting': self._generate_troubleshooting_guides
        }

    def enhance_documentation(self) -> Dict[str, Any]:
        """执行文档完善"""
        logger.info("开始文档完善...")

        # 分析现有文档
        existing_docs = self._analyze_existing_documentation()

        # 生成新文档
        documentation_results = {}

        for doc_type, doc_generator in self.documentation_types.items():
            try:
                result = doc_generator(existing_docs)
                documentation_results[doc_type] = result
                logger.info(f"{doc_type}文档生成完成")
            except Exception as e:
                logger.error(f"{doc_type}文档生成失败: {e}")
                documentation_results[doc_type] = {'error': str(e)}

        # 保存文档日志
        self._save_documentation_log(existing_docs, documentation_results)

        return {
            'existing_docs': existing_docs,
            'documentation_results': documentation_results
        }

    def _analyze_existing_documentation(self) -> Dict[str, Any]:
        """分析现有文档"""
        logger.info("分析现有文档...")

        docs_analysis = {
            'timestamp': datetime.now().isoformat(),
            'existing_files': [],
            'missing_docs': [],
            'outdated_docs': [],
            'coverage_analysis': {}
        }

        # 扫描现有文档
        for doc_file in self.docs_dir.rglob("*.md"):
            docs_analysis['existing_files'].append(str(doc_file))

        # 检查缺失的文档
        required_docs = [
            'api/README.md',
            'architecture/infrastructure/API_REFERENCE.md',
            'guides/INFRASTRUCTURE_USAGE_GUIDE.md',
            'deployment/INFRASTRUCTURE_DEPLOYMENT.md',
            'troubleshooting/INFRASTRUCTURE_TROUBLESHOOTING.md'
        ]

        for required_doc in required_docs:
            doc_path = self.docs_dir / required_doc
            if not doc_path.exists():
                docs_analysis['missing_docs'].append(required_doc)

        # 分析代码覆盖率
        docs_analysis['coverage_analysis'] = self._analyze_code_coverage()

        return docs_analysis

    def _analyze_code_coverage(self) -> Dict[str, Any]:
        """分析代码文档覆盖率"""
        coverage = {
            'total_modules': 0,
            'documented_modules': 0,
            'total_classes': 0,
            'documented_classes': 0,
            'total_functions': 0,
            'documented_functions': 0
        }

        # 分析基础设施层代码
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析Python代码
                tree = ast.parse(content)

                # 统计模块、类、函数
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        coverage['total_classes'] += 1
                        if node.docstring:
                            coverage['documented_classes'] += 1
                    elif isinstance(node, ast.FunctionDef):
                        coverage['total_functions'] += 1
                        if node.docstring:
                            coverage['documented_functions'] += 1

                coverage['total_modules'] += 1
                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    coverage['documented_modules'] += 1

            except Exception:
                continue

        return coverage

    def _generate_api_documentation(self, existing_docs: Dict[str, Any]) -> Dict[str, Any]:
        """生成API文档"""
        logger.info("生成API文档...")

        api_docs = []

        # 扫描基础设施层模块
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                module_doc = self._extract_module_documentation(py_file)
                if module_doc:
                    api_docs.append(module_doc)
            except Exception as e:
                logger.warning(f"无法解析模块 {py_file}: {e}")

        # 生成API参考文档
        api_reference_path = self.docs_dir / "architecture" / "infrastructure" / "API_REFERENCE.md"
        api_reference_path.parent.mkdir(parents=True, exist_ok=True)

        api_content = self._generate_api_reference_content(api_docs)
        with open(api_reference_path, 'w', encoding='utf-8') as f:
            f.write(api_content)

        return {
            'api_docs_generated': True,
            'api_reference_path': str(api_reference_path),
            'modules_documented': len(api_docs)
        }

    def _extract_module_documentation(self, py_file: Path) -> Optional[Dict[str, Any]]:
        """提取模块文档"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            module_info = {
                'name': py_file.stem,
                'path': str(py_file.relative_to(self.project_root)),
                'classes': [],
                'functions': [],
                'docstring': ast.get_docstring(tree)
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': []
                    }

                    for child in ast.walk(node):
                        if isinstance(child, ast.FunctionDef):
                            method_info = {
                                'name': child.name,
                                'docstring': ast.get_docstring(child),
                                'args': [arg.arg for arg in child.args.args]
                            }
                            class_info['methods'].append(method_info)

                    module_info['classes'].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    function_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args]
                    }
                    module_info['functions'].append(function_info)

            return module_info

        except Exception as e:
            logger.warning(f"无法解析模块 {py_file}: {e}")
            return None

    def _generate_api_reference_content(self, api_docs: List[Dict[str, Any]]) -> str:
        """生成API参考文档内容"""
        content = []
        content.append("# 基础设施层 API 参考文档")
        content.append(f"生成时间: {datetime.now().isoformat()}")
        content.append("")
        content.append("## 概述")
        content.append("本文档提供了RQA2025系统基础设施层的完整API参考。")
        content.append("")

        # 按模块组织API文档
        for module_doc in api_docs:
            content.append(f"## {module_doc['name']}")
            content.append(f"**文件路径**: `{module_doc['path']}`")
            content.append("")

            if module_doc['docstring']:
                content.append("### 模块描述")
                content.append(module_doc['docstring'])
                content.append("")

            # 类文档
            if module_doc['classes']:
                content.append("### 类")
                for class_info in module_doc['classes']:
                    content.append(f"#### {class_info['name']}")
                    if class_info['docstring']:
                        content.append(class_info['docstring'])
                    content.append("")

                    if class_info['methods']:
                        content.append("**方法:**")
                        for method in class_info['methods']:
                            content.append(f"- `{method['name']}({', '.join(method['args'])})`")
                            if method['docstring']:
                                content.append(f"  - {method['docstring']}")
                        content.append("")

            # 函数文档
            if module_doc['functions']:
                content.append("### 函数")
                for func_info in module_doc['functions']:
                    content.append(f"#### {func_info['name']}")
                    content.append(f"```python")
                    content.append(f"{func_info['name']}({', '.join(func_info['args'])})")
                    content.append(f"```")
                    if func_info['docstring']:
                        content.append(func_info['docstring'])
                    content.append("")

        return "\n".join(content)

    def _generate_usage_guides(self, existing_docs: Dict[str, Any]) -> Dict[str, Any]:
        """生成使用指南"""
        logger.info("生成使用指南...")

        # 生成基础设施层使用指南
        usage_guide_path = self.docs_dir / "guides" / "INFRASTRUCTURE_USAGE_GUIDE.md"
        usage_guide_path.parent.mkdir(parents=True, exist_ok=True)

        usage_content = self._generate_usage_guide_content()
        with open(usage_guide_path, 'w', encoding='utf-8') as f:
            f.write(usage_content)

        return {
            'usage_guide_generated': True,
            'usage_guide_path': str(usage_guide_path)
        }

    def _generate_usage_guide_content(self) -> str:
        """生成使用指南内容"""
        content = []
        content.append("# 基础设施层使用指南")
        content.append(f"生成时间: {datetime.now().isoformat()}")
        content.append("")

        content.append("## 概述")
        content.append("基础设施层为RQA2025系统提供核心的基础服务，包括配置管理、日志管理、监控系统等。")
        content.append("")

        content.append("## 快速开始")
        content.append("")
        content.append("### 1. 配置管理")
        content.append("```python")
        content.append("from src.infrastructure.config import ConfigManager")
        content.append("")
        content.append("# 创建配置管理器")
        content.append("config_manager = ConfigManager('config')")
        content.append("")
        content.append("# 获取配置")
        content.append("database_config = config_manager.get_config('database')")
        content.append("```")
        content.append("")

        content.append("### 2. 日志管理")
        content.append("```python")
        content.append("from src.infrastructure.logging import get_unified_logger")
        content.append("")
        content.append("# 获取日志器")
        content.append("logger = get_unified_logger('my_module')")
        content.append("")
        content.append("# 记录日志")
        content.append("logger.info('操作成功')")
        content.append("logger.error('发生错误')")
        content.append("```")
        content.append("")

        content.append("### 3. 监控系统")
        content.append("```python")
        content.append("from src.infrastructure.monitoring import AutomationMonitor")
        content.append("")
        content.append("# 创建监控器")
        content.append("monitor = AutomationMonitor()")
        content.append("")
        content.append("# 记录指标")
        content.append("monitor.record_metric('cpu_usage', 75.5)")
        content.append("")
        content.append("# 获取指标")
        content.append("metrics = monitor.get_metric('cpu_usage')")
        content.append("```")
        content.append("")

        content.append("## 高级用法")
        content.append("")
        content.append("### 自定义配置")
        content.append("```python")
        content.append("# 自定义配置路径")
        content.append("config_manager = ConfigManager('custom_config_path')")
        content.append("")
        content.append("# 热重载配置")
        content.append("config_manager.enable_hot_reload()")
        content.append("```")
        content.append("")

        content.append("### 结构化日志")
        content.append("```python")
        content.append("from src.infrastructure.logging import EnhancedLogManager")
        content.append("")
        content.append("log_manager = EnhancedLogManager()")
        content.append("log_manager.log_structured('business_event', {")
        content.append("    'user_id': 123,")
        content.append("    'action': 'login',")
        content.append("    'timestamp': '2025-08-06T14:00:00Z'")
        content.append("})")
        content.append("```")
        content.append("")

        content.append("## 最佳实践")
        content.append("")
        content.append("1. **配置管理**: 使用环境变量覆盖默认配置")
        content.append("2. **日志记录**: 使用结构化日志便于分析")
        content.append("3. **监控指标**: 定期清理历史数据")
        content.append("4. **错误处理**: 实现优雅的错误恢复机制")
        content.append("5. **性能优化**: 使用缓存减少重复计算")
        content.append("")

        content.append("## 故障排除")
        content.append("")
        content.append("### 常见问题")
        content.append("")
        content.append("**Q: 配置加载失败怎么办？**")
        content.append("A: 检查配置文件路径和格式，确保JSON/YAML语法正确")
        content.append("")
        content.append("**Q: 日志文件过大怎么办？**")
        content.append("A: 配置日志轮转策略，定期清理旧日志文件")
        content.append("")
        content.append("**Q: 监控指标不准确怎么办？**")
        content.append("A: 检查指标收集逻辑，确保数据源正确")

        return "\n".join(content)

    def _generate_architecture_documentation(self, existing_docs: Dict[str, Any]) -> Dict[str, Any]:
        """生成架构文档"""
        logger.info("生成架构文档...")

        # 生成架构概览文档
        arch_overview_path = self.docs_dir / "architecture" / "infrastructure" / "ARCHITECTURE_OVERVIEW.md"
        arch_overview_path.parent.mkdir(parents=True, exist_ok=True)

        arch_content = self._generate_architecture_content()
        with open(arch_overview_path, 'w', encoding='utf-8') as f:
            f.write(arch_content)

        return {
            'architecture_docs_generated': True,
            'architecture_overview_path': str(arch_overview_path)
        }

    def _generate_architecture_content(self) -> str:
        """生成架构文档内容"""
        content = []
        content.append("# 基础设施层架构概览")
        content.append(f"生成时间: {datetime.now().isoformat()}")
        content.append("")

        content.append("## 架构设计原则")
        content.append("")
        content.append("### 1. 高可用性")
        content.append("- 服务冗余和故障转移")
        content.append("- 健康检查和自动恢复")
        content.append("- 负载均衡和流量控制")
        content.append("")

        content.append("### 2. 可扩展性")
        content.append("- 水平扩展支持")
        content.append("- 模块化设计")
        content.append("- 插件化架构")
        content.append("")

        content.append("### 3. 易维护性")
        content.append("- 清晰的代码结构")
        content.append("- 完善的文档")
        content.append("- 自动化测试")
        content.append("")

        content.append("### 4. 易集成性")
        content.append("- 标准化接口")
        content.append("- 多种集成方式")
        content.append("- 向后兼容")
        content.append("")

        content.append("## 核心组件")
        content.append("")
        content.append("### 配置管理 (Config Management)")
        content.append("- 统一配置接口")
        content.append("- 多环境支持")
        content.append("- 热重载功能")
        content.append("- 配置验证")
        content.append("")

        content.append("### 日志管理 (Logging Management)")
        content.append("- 统一日志接口")
        content.append("- 结构化日志")
        content.append("- 日志聚合")
        content.append("- 性能监控")
        content.append("")

        content.append("### 监控系统 (Monitoring System)")
        content.append("- 实时监控")
        content.append("- 告警管理")
        content.append("- 指标收集")
        content.append("- 可视化面板")
        content.append("")

        content.append("### 安全模块 (Security Module)")
        content.append("- 认证授权")
        content.append("- 数据加密")
        content.append("- 访问控制")
        content.append("- 审计日志")
        content.append("")

        content.append("## 技术栈")
        content.append("")
        content.append("### 核心框架")
        content.append("- Python 3.9+")
        content.append("- asyncio (异步编程)")
        content.append("- dataclasses (数据类)")
        content.append("- typing (类型注解)")
        content.append("")

        content.append("### 数据存储")
        content.append("- Redis (缓存)")
        content.append("- PostgreSQL (关系数据库)")
        content.append("- InfluxDB (时序数据库)")
        content.append("- 文件系统 (配置存储)")
        content.append("")

        content.append("### 监控工具")
        content.append("- Prometheus (指标收集)")
        content.append("- Grafana (可视化)")
        content.append("- AlertManager (告警)")
        content.append("- 自定义监控器")
        content.append("")

        content.append("## 部署架构")
        content.append("")
        content.append("### 单机部署")
        content.append("适用于开发和测试环境")
        content.append("")
        content.append("### 集群部署")
        content.append("适用于生产环境，支持高可用")
        content.append("")
        content.append("### 云原生部署")
        content.append("支持Kubernetes和Docker")

        return "\n".join(content)

    def _generate_deployment_guides(self, existing_docs: Dict[str, Any]) -> Dict[str, Any]:
        """生成部署指南"""
        logger.info("生成部署指南...")

        # 生成部署指南
        deployment_guide_path = self.docs_dir / "deployment" / "INFRASTRUCTURE_DEPLOYMENT.md"
        deployment_guide_path.parent.mkdir(parents=True, exist_ok=True)

        deployment_content = self._generate_deployment_guide_content()
        with open(deployment_guide_path, 'w', encoding='utf-8') as f:
            f.write(deployment_content)

        return {
            'deployment_guide_generated': True,
            'deployment_guide_path': str(deployment_guide_path)
        }

    def _generate_deployment_guide_content(self) -> str:
        """生成部署指南内容"""
        content = []
        content.append("# 基础设施层部署指南")
        content.append(f"生成时间: {datetime.now().isoformat()}")
        content.append("")

        content.append("## 环境要求")
        content.append("")
        content.append("### 系统要求")
        content.append("- Python 3.9 或更高版本")
        content.append("- 内存: 最少 4GB，推荐 8GB")
        content.append("- 存储: 最少 10GB 可用空间")
        content.append("- 网络: 稳定的网络连接")
        content.append("")

        content.append("### 依赖服务")
        content.append("- Redis 6.0+")
        content.append("- PostgreSQL 12+")
        content.append("- InfluxDB 2.0+ (可选)")
        content.append("")

        content.append("## 安装步骤")
        content.append("")
        content.append("### 1. 克隆代码")
        content.append("```bash")
        content.append("git clone <repository_url>")
        content.append("cd RQA2025")
        content.append("```")
        content.append("")

        content.append("### 2. 安装依赖")
        content.append("```bash")
        content.append("pip install -r requirements.txt")
        content.append("```")
        content.append("")

        content.append("### 3. 配置环境")
        content.append("```bash")
        content.append("# 复制配置模板")
        content.append("cp config/config.template.json config/config.json")
        content.append("")
        content.append("# 编辑配置文件")
        content.append("vim config/config.json")
        content.append("```")
        content.append("")

        content.append("### 4. 初始化数据库")
        content.append("```bash")
        content.append("python scripts/setup_database.py")
        content.append("```")
        content.append("")

        content.append("### 5. 启动服务")
        content.append("```bash")
        content.append("python scripts/start_infrastructure.py")
        content.append("```")
        content.append("")

        content.append("## 配置说明")
        content.append("")
        content.append("### 数据库配置")
        content.append("```json")
        content.append("{")
        content.append('  "database": {')
        content.append('    "host": "localhost",')
        content.append('    "port": 5432,')
        content.append('    "name": "rqa2025",')
        content.append('    "user": "rqa_user",')
        content.append('    "password": "secure_password"')
        content.append("  }")
        content.append("}")
        content.append("```")
        content.append("")

        content.append("### Redis配置")
        content.append("```json")
        content.append("{")
        content.append('  "redis": {')
        content.append('    "host": "localhost",')
        content.append('    "port": 6379,')
        content.append('    "db": 0,')
        content.append('    "password": null')
        content.append("  }")
        content.append("}")
        content.append("```")
        content.append("")

        content.append("## 生产环境部署")
        content.append("")
        content.append("### 使用Docker")
        content.append("```bash")
        content.append("# 构建镜像")
        content.append("docker build -t rqa2025-infrastructure .")
        content.append("")
        content.append("# 运行容器")
        content.append("docker run -d --name rqa-infrastructure \\")
        content.append("  -p 8080:8080 \\")
        content.append("  -v /path/to/config:/app/config \\")
        content.append("  rqa2025-infrastructure")
        content.append("```")
        content.append("")

        content.append("### 使用Kubernetes")
        content.append("```yaml")
        content.append("apiVersion: apps/v1")
        content.append("kind: Deployment")
        content.append("metadata:")
        content.append("  name: rqa-infrastructure")
        content.append("spec:")
        content.append("  replicas: 3")
        content.append("  selector:")
        content.append("    matchLabels:")
        content.append("      app: rqa-infrastructure")
        content.append("  template:")
        content.append("    metadata:")
        content.append("      labels:")
        content.append("        app: rqa-infrastructure")
        content.append("    spec:")
        content.append("      containers:")
        content.append("      - name: infrastructure")
        content.append("        image: rqa2025-infrastructure:latest")
        content.append("        ports:")
        content.append("        - containerPort: 8080")
        content.append("```")
        content.append("")

        content.append("## 监控和日志")
        content.append("")
        content.append("### 健康检查")
        content.append("```bash")
        content.append("curl http://localhost:8080/health")
        content.append("```")
        content.append("")

        content.append("### 查看日志")
        content.append("```bash")
        content.append("tail -f logs/infrastructure.log")
        content.append("```")
        content.append("")

        content.append("### 监控指标")
        content.append("```bash")
        content.append("curl http://localhost:8080/metrics")
        content.append("```")

        return "\n".join(content)

    def _generate_troubleshooting_guides(self, existing_docs: Dict[str, Any]) -> Dict[str, Any]:
        """生成故障排除指南"""
        logger.info("生成故障排除指南...")

        # 生成故障排除指南
        troubleshooting_guide_path = self.docs_dir / "troubleshooting" / "INFRASTRUCTURE_TROUBLESHOOTING.md"
        troubleshooting_guide_path.parent.mkdir(parents=True, exist_ok=True)

        troubleshooting_content = self._generate_troubleshooting_content()
        with open(troubleshooting_guide_path, 'w', encoding='utf-8') as f:
            f.write(troubleshooting_content)

        return {
            'troubleshooting_guide_generated': True,
            'troubleshooting_guide_path': str(troubleshooting_guide_path)
        }

    def _generate_troubleshooting_content(self) -> str:
        """生成故障排除指南内容"""
        content = []
        content.append("# 基础设施层故障排除指南")
        content.append(f"生成时间: {datetime.now().isoformat()}")
        content.append("")

        content.append("## 常见问题")
        content.append("")

        content.append("### 1. 配置加载失败")
        content.append("**症状**: 应用启动时出现配置错误")
        content.append("")
        content.append("**可能原因**:")
        content.append("- 配置文件路径错误")
        content.append("- 配置文件格式错误")
        content.append("- 权限不足")
        content.append("")
        content.append("**解决方案**:")
        content.append("```bash")
        content.append("# 检查配置文件")
        content.append("ls -la config/")
        content.append("")
        content.append("# 验证JSON格式")
        content.append("python -m json.tool config/config.json")
        content.append("")
        content.append("# 检查权限")
        content.append("chmod 644 config/config.json")
        content.append("```")
        content.append("")

        content.append("### 2. 数据库连接失败")
        content.append("**症状**: 无法连接到数据库")
        content.append("")
        content.append("**可能原因**:")
        content.append("- 数据库服务未启动")
        content.append("- 连接参数错误")
        content.append("- 网络问题")
        content.append("")
        content.append("**解决方案**:")
        content.append("```bash")
        content.append("# 检查数据库服务")
        content.append("systemctl status postgresql")
        content.append("")
        content.append("# 测试连接")
        content.append("psql -h localhost -U rqa_user -d rqa2025")
        content.append("")
        content.append("# 检查网络")
        content.append("telnet localhost 5432")
        content.append("```")
        content.append("")

        content.append("### 3. Redis连接失败")
        content.append("**症状**: 缓存功能不可用")
        content.append("")
        content.append("**可能原因**:")
        content.append("- Redis服务未启动")
        content.append("- 端口被占用")
        content.append("- 内存不足")
        content.append("")
        content.append("**解决方案**:")
        content.append("```bash")
        content.append("# 检查Redis服务")
        content.append("systemctl status redis")
        content.append("")
        content.append("# 测试连接")
        content.append("redis-cli ping")
        content.append("")
        content.append("# 检查内存")
        content.append("free -h")
        content.append("```")
        content.append("")

        content.append("### 4. 日志文件过大")
        content.append("**症状**: 磁盘空间不足")
        content.append("")
        content.append("**解决方案**:")
        content.append("```bash")
        content.append("# 查看日志文件大小")
        content.append("du -sh logs/*")
        content.append("")
        content.append("# 清理旧日志")
        content.append("find logs/ -name '*.log.*' -mtime +7 -delete")
        content.append("")
        content.append("# 配置日志轮转")
        content.append("vim config/logging.json")
        content.append("```")
        content.append("")

        content.append("### 5. 性能问题")
        content.append("**症状**: 响应时间过长")
        content.append("")
        content.append("**诊断步骤**:")
        content.append("```bash")
        content.append("# 检查CPU使用率")
        content.append("top")
        content.append("")
        content.append("# 检查内存使用")
        content.append("free -h")
        content.append("")
        content.append("# 检查磁盘I/O")
        content.append("iostat -x 1")
        content.append("")
        content.append("# 检查网络")
        content.append("netstat -i")
        content.append("```")
        content.append("")

        content.append("## 调试工具")
        content.append("")
        content.append("### 1. 日志分析")
        content.append("```bash")
        content.append("# 查看错误日志")
        content.append("grep ERROR logs/infrastructure.log")
        content.append("")
        content.append("# 查看特定时间段的日志")
        content.append("sed -n '/2025-08-06 14:00/,/2025-08-06 15:00/p' logs/infrastructure.log")
        content.append("")
        content.append("# 统计错误次数")
        content.append("grep -c ERROR logs/infrastructure.log")
        content.append("```")
        content.append("")

        content.append("### 2. 性能分析")
        content.append("```bash")
        content.append("# 使用cProfile分析性能")
        content.append("python -m cProfile -o profile.stats main.py")
        content.append("")
        content.append("# 分析结果")
        content.append(
            "python -c \"import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)\"")
        content.append("```")
        content.append("")

        content.append("### 3. 内存分析")
        content.append("```bash")
        content.append("# 使用memory_profiler")
        content.append("pip install memory_profiler")
        content.append("python -m memory_profiler script.py")
        content.append("```")
        content.append("")

        content.append("## 联系支持")
        content.append("")
        content.append("如果以上解决方案无法解决问题，请联系技术支持：")
        content.append("- 邮箱: support@rqa2025.com")
        content.append("- 电话: +86-400-123-4567")
        content.append("- 在线文档: https://docs.rqa2025.com")

        return "\n".join(content)

    def _save_documentation_log(self, existing_docs: Dict[str, Any], documentation_results: Dict[str, Any]):
        """保存文档日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'existing_docs': existing_docs,
            'documentation_results': documentation_results
        }

        with open(self.documentation_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.info(f"文档日志已保存到: {self.documentation_log}")

    def generate_documentation_report(self) -> str:
        """生成文档报告"""
        if not self.documentation_log.exists():
            return "未找到文档日志，请先运行文档完善"

        with open(self.documentation_log, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        report = []
        report.append("# 基础设施层文档完善报告")
        report.append(f"生成时间: {log_data['timestamp']}")
        report.append("")

        # 现有文档分析
        existing_docs = log_data['existing_docs']
        report.append("## 现有文档分析")
        report.append(f"- 现有文档文件数: {len(existing_docs['existing_files'])}")
        report.append(f"- 缺失文档数: {len(existing_docs['missing_docs'])}")
        report.append(f"- 过时文档数: {len(existing_docs['outdated_docs'])}")
        report.append("")

        # 代码覆盖率
        coverage = existing_docs['coverage_analysis']
        report.append("## 代码文档覆盖率")
        report.append(
            f"- 模块覆盖率: {coverage['documented_modules']}/{coverage['total_modules']} ({coverage['documented_modules']/max(coverage['total_modules'], 1)*100:.1f}%)")
        report.append(
            f"- 类覆盖率: {coverage['documented_classes']}/{coverage['total_classes']} ({coverage['documented_classes']/max(coverage['total_classes'], 1)*100:.1f}%)")
        report.append(
            f"- 函数覆盖率: {coverage['documented_functions']}/{coverage['total_functions']} ({coverage['documented_functions']/max(coverage['total_functions'], 1)*100:.1f}%)")
        report.append("")

        # 文档生成结果
        documentation_results = log_data['documentation_results']
        report.append("## 文档生成结果")
        for doc_type, result in documentation_results.items():
            if 'error' not in result:
                report.append(f"- {doc_type}: ✅ 生成成功")
            else:
                report.append(f"- {doc_type}: ❌ 生成失败 - {result['error']}")

        return "\n".join(report)


def main():
    """主函数"""
    project_root = Path.cwd()
    enhancer = InfrastructureDocumentationEnhancer(str(project_root))

    # 执行文档完善
    results = enhancer.enhance_documentation()

    # 生成报告
    report = enhancer.generate_documentation_report()
    print(report)

    # 保存报告
    report_path = project_root / "reports" / "infrastructure_documentation_enhancement_report.md"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n文档报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
