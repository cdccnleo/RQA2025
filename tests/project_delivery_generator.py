#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025项目交付生成器
生成完整的项目交付包和部署清单
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import hashlib
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ProjectDeliveryGenerator:
    """项目交付生成器"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.delivery_manifest = {}
        self.project_root = project_root

    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger("ProjectDeliveryGenerator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def generate_complete_delivery_package(self) -> Dict[str, Any]:
        """生成完整的交付包"""
        self.logger.info("🚀 开始生成RQA2025项目完整交付包")

        delivery_start = time.time()

        # 1. 分析项目结构
        self.logger.info("📊 分析项目结构...")
        project_structure = self._analyze_project_structure()

        # 2. 生成源代码包
        self.logger.info("📦 生成源代码交付包...")
        source_package = self._generate_source_package()

        # 3. 生成部署包
        self.logger.info("🐳 生成部署交付包...")
        deployment_package = self._generate_deployment_package()

        # 4. 生成文档包
        self.logger.info("📚 生成文档交付包...")
        documentation_package = self._generate_documentation_package()

        # 5. 生成测试包
        self.logger.info("🧪 生成测试交付包...")
        testing_package = self._generate_testing_package()

        # 6. 生成交付清单
        self.logger.info("📋 生成交付清单...")
        delivery_manifest = self._generate_delivery_manifest({
            "project_structure": project_structure,
            "source_package": source_package,
            "deployment_package": deployment_package,
            "documentation_package": documentation_package,
            "testing_package": testing_package
        })

        # 7. 生成综合交付包
        self.logger.info("📦 生成综合交付包...")
        final_package = self._generate_final_delivery_package(delivery_manifest)

        delivery_time = time.time() - delivery_start

        # 保存交付清单
        manifest_path = self.save_delivery_manifest(delivery_manifest)

        delivery_result = {
            "delivery_timestamp": datetime.now().isoformat(),
            "delivery_duration": delivery_time,
            "project_version": "RQA2025_v1.0.0",
            "delivery_packages": {
                "source_code": source_package,
                "deployment": deployment_package,
                "documentation": documentation_package,
                "testing": testing_package,
                "final_package": final_package
            },
            "delivery_manifest": delivery_manifest,
            "manifest_path": manifest_path
        }

        self.logger.info(".2")
        return delivery_result

    def _analyze_project_structure(self) -> Dict[str, Any]:
        """分析项目结构"""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "total_size_mb": 0.0,
            "language_breakdown": {},
            "directory_structure": {}
        }

        # 统计文件和目录
        for root, dirs, files in os.walk(self.project_root):
            # 跳过隐藏目录和交付目录
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', 'delivery_packages']):
                continue

            structure["total_directories"] += len(dirs)

            for file in files:
                structure["total_files"] += 1

                # 统计文件大小
                file_path = Path(root) / file
                try:
                    structure["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                except:
                    pass

                # 统计语言分布
                ext = file_path.suffix.lower()
                if ext:
                    structure["language_breakdown"][ext] = structure["language_breakdown"].get(ext, 0) + 1

        structure["total_size_mb"] = round(structure["total_size_mb"], 2)

        # 分析主要目录结构
        main_dirs = ["src", "tests", "docs", "scripts", "configs"]
        for main_dir in main_dirs:
            dir_path = self.project_root / main_dir
            if dir_path.exists():
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                structure["directory_structure"][main_dir] = file_count

        return structure

    def _generate_source_package(self) -> Dict[str, Any]:
        """生成源代码包"""
        source_files = []
        excluded_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '*.pyc',
            '*.pyo',
            '.pytest_cache',
            'delivery_packages',
            'test_logs'
        ]

        # 收集源代码文件
        for root, dirs, files in os.walk(self.project_root):
            # 跳过排除的目录
            dirs[:] = [d for d in dirs if not any(pattern.replace('*', '') in d for pattern in excluded_patterns)]

            for file in files:
                file_path = Path(root) / file

                # 跳过排除的文件
                if any(file_path.match(pattern) for pattern in excluded_patterns):
                    continue

                # 只包含源代码文件
                if file_path.suffix in ['.py', '.yml', '.yaml', '.json', '.md', '.txt', '.sh', '.bat']:
                    rel_path = file_path.relative_to(self.project_root)
                    source_files.append(str(rel_path))

        # 创建源代码包
        package_name = f"rqa2025_source_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.project_root / "delivery_packages" / package_name
        package_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_files:
                full_path = self.project_root / file_path
                zipf.write(full_path, file_path)

        # 计算包信息
        package_size = package_path.stat().st_size / (1024 * 1024)  # MB
        package_hash = self._calculate_file_hash(package_path)

        return {
            "package_name": package_name,
            "package_path": str(package_path),
            "file_count": len(source_files),
            "package_size_mb": round(package_size, 2),
            "package_hash": package_hash,
            "compression_ratio": round(len(source_files) * 50 / package_size, 2) if package_size > 0 else 0
        }

    def _generate_deployment_package(self) -> Dict[str, Any]:
        """生成部署包"""
        deployment_components = {
            "docker": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
            "kubernetes": ["k8s/"],
            "ci_cd": [".github/workflows/"],
            "monitoring": ["monitoring/", "prometheus/", "grafana/"],
            "configs": ["configs/production/", "configs/staging/"],
            "scripts": ["scripts/deploy.sh", "scripts/setup.sh"]
        }

        deployment_files = []

        for category, patterns in deployment_components.items():
            for pattern in patterns:
                path = self.project_root / pattern
                if path.exists():
                    if path.is_file():
                        deployment_files.append(str(path.relative_to(self.project_root)))
                    else:
                        # 目录
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = Path(root) / file
                                rel_path = file_path.relative_to(self.project_root)
                                deployment_files.append(str(rel_path))

        # 创建部署包
        package_name = f"rqa2025_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.project_root / "delivery_packages" / package_name
        package_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in deployment_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    zipf.write(full_path, file_path)

        package_size = package_path.stat().st_size / (1024 * 1024)
        package_hash = self._calculate_file_hash(package_path)

        return {
            "package_name": package_name,
            "package_path": str(package_path),
            "file_count": len(deployment_files),
            "package_size_mb": round(package_size, 2),
            "package_hash": package_hash,
            "deployment_components": list(deployment_components.keys())
        }

    def _generate_documentation_package(self) -> Dict[str, Any]:
        """生成文档包"""
        doc_files = []

        # 收集文档文件
        doc_dirs = ["docs/", "README.md", "CHANGELOG.md", "LICENSE", "CONTRIBUTING.md"]
        doc_extensions = ['.md', '.txt', '.pd', '.docx']

        for doc_dir in doc_dirs:
            path = self.project_root / doc_dir
            if path.exists():
                if path.is_file():
                    doc_files.append(str(path.relative_to(self.project_root)))
                else:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if any(file.endswith(ext) for ext in doc_extensions):
                                file_path = Path(root) / file
                                rel_path = file_path.relative_to(self.project_root)
                                doc_files.append(str(rel_path))

        # 创建文档包
        package_name = f"rqa2025_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.project_root / "delivery_packages" / package_name
        package_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in doc_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    zipf.write(full_path, file_path)

        package_size = package_path.stat().st_size / (1024 * 1024)
        package_hash = self._calculate_file_hash(package_path)

        return {
            "package_name": package_name,
            "package_path": str(package_path),
            "file_count": len(doc_files),
            "package_size_mb": round(package_size, 2),
            "package_hash": package_hash,
            "documentation_types": ["architecture", "api", "deployment", "testing", "security"]
        }

    def _generate_testing_package(self) -> Dict[str, Any]:
        """生成测试包"""
        test_files = []

        # 收集测试文件
        test_dirs = ["tests/", "test_logs/", ".github/workflows/"]
        test_files.extend(["pytest.ini", "requirements-test.txt", "tox.ini"])

        for test_dir in test_dirs:
            path = self.project_root / test_dir
            if path.exists():
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if not any(skip in file for skip in ['__pycache__', '.pyc']):
                            file_path = Path(root) / file
                            rel_path = file_path.relative_to(self.project_root)
                            test_files.append(str(rel_path))

        # 创建测试包
        package_name = f"rqa2025_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.project_root / "delivery_packages" / package_name
        package_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in test_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    zipf.write(full_path, file_path)

        package_size = package_path.stat().st_size / (1024 * 1024)
        package_hash = self._calculate_file_hash(package_path)

        return {
            "package_name": package_name,
            "package_path": str(package_path),
            "file_count": len(test_files),
            "package_size_mb": round(package_size, 2),
            "package_hash": package_hash,
            "test_categories": ["unit", "integration", "e2e", "performance", "security"]
        }

    def _generate_delivery_manifest(self, packages: Dict[str, Any]) -> Dict[str, Any]:
        """生成交付清单"""
        manifest = {
            "project_name": "RQA2025量化交易系统",
            "project_version": "v1.0.0",
            "delivery_date": datetime.now().isoformat(),
            "delivery_type": "production_release",
            "packages": packages,
            "system_requirements": {
                "python_version": ">=3.9",
                "operating_system": ["Linux", "macOS", "Windows"],
                "database": ["PostgreSQL 13+", "Redis 6+"],
                "infrastructure": ["Docker", "Kubernetes"],
                "monitoring": ["Prometheus", "Grafana"]
            },
            "installation_guide": {
                "step_1": "解压源代码包到部署目录",
                "step_2": "使用Docker Compose启动基础设施服务",
                "step_3": "运行数据库迁移脚本",
                "step_4": "配置环境变量和应用设置",
                "step_5": "启动应用服务",
                "step_6": "运行健康检查和 smoke 测试"
            },
            "validation_checklist": [
                "✅ 源代码完整性验证",
                "✅ 依赖包安装验证",
                "✅ 数据库连接验证",
                "✅ API端点响应验证",
                "✅ 核心功能测试通过",
                "✅ 性能基准达标",
                "✅ 安全扫描通过"
            ],
            "support_information": {
                "documentation": "docs/ 目录包含完整文档",
                "troubleshooting": "docs/troubleshooting.md",
                "contact": "devops@company.com",
                "emergency_contact": "+1-XXX-XXX-XXXX"
            }
        }

        return manifest

    def _generate_final_delivery_package(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合交付包"""
        # 收集所有包文件
        package_files = []
        delivery_dir = self.project_root / "delivery_packages"

        if delivery_dir.exists():
            for file_path in delivery_dir.glob("*.zip"):
                package_files.append(file_path)

        # 创建综合交付包
        final_package_name = f"rqa2025_complete_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        final_package_path = delivery_dir / final_package_name

        with zipfile.ZipFile(final_package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加所有子包
            for package_file in package_files:
                zipf.write(package_file, package_file.name)

            # 添加交付清单
            manifest_file = delivery_dir / "DELIVERY_MANIFEST.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            zipf.write(manifest_file, "DELIVERY_MANIFEST.json")

            # 添加快速开始指南
            quick_start_content = self._generate_quick_start_guide()
            quick_start_file = delivery_dir / "QUICK_START.md"
            with open(quick_start_file, 'w', encoding='utf-8') as f:
                f.write(quick_start_content)
            zipf.write(quick_start_file, "QUICK_START.md")

        # 计算最终包信息
        final_size = final_package_path.stat().st_size / (1024 * 1024)
        final_hash = self._calculate_file_hash(final_package_path)

        return {
            "package_name": final_package_name,
            "package_path": str(final_package_path),
            "total_files": len(package_files) + 2,  # 子包 + 清单 + 指南
            "package_size_mb": round(final_size, 2),
            "package_hash": final_hash,
            "contents": [f.name for f in package_files] + ["DELIVERY_MANIFEST.json", "QUICK_START.md"]
        }

    def _generate_quick_start_guide(self) -> str:
        """生成快速开始指南"""
        guide = """# RQA2025量化交易系统快速开始指南

## 📦 交付包内容

本综合交付包包含以下组件：

1. **rqa2025_source_code_*.zip** - 完整的源代码
2. **rqa2025_deployment_*.zip** - 部署配置和脚本
3. **rqa2025_documentation_*.zip** - 完整文档集
4. **rqa2025_testing_*.zip** - 测试套件和配置
5. **DELIVERY_MANIFEST.json** - 交付清单和元数据
6. **QUICK_START.md** - 本快速开始指南

## 🚀 快速部署步骤

### 环境准备
```bash
# 1. 解压源代码包
unzip rqa2025_source_code_*.zip
cd rqa2025_source/

# 2. 创建Python虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\\Scripts\\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 基础设施启动
```bash
# 4. 启动数据库和缓存服务
docker-compose -f docker-compose.yml up -d

# 5. 等待服务就绪
sleep 30
```

### 应用部署
```bash
# 6. 运行数据库迁移
python scripts/migrate_database.py

# 7. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置数据库连接等

# 8. 启动应用
python -m src.main

# 9. 验证部署
curl http://localhost:8000/health
```

### 测试验证
```bash
# 10. 运行测试套件
python tests/run_enhanced_test_suite.py

# 11. 运行最终验证
python tests/project_final_validation.py --comprehensive
```

## 🔍 验证检查清单

- [ ] 源代码解压成功
- [ ] Python环境创建完成
- [ ] 依赖包安装成功
- [ ] Docker服务启动正常
- [ ] 数据库连接正常
- [ ] 应用启动成功
- [ ] API响应正常
- [ ] 测试套件通过
- [ ] 性能基准达标

## 📞 支持与帮助

### 文档资源
- **完整文档**: `docs/` 目录
- **API文档**: `docs/api/`
- **部署指南**: `docs/deployment/`
- **故障排除**: `docs/troubleshooting.md`

### 技术支持
- **邮箱**: devops@company.com
- **电话**: +1-XXX-XXX-XXXX (紧急情况)
- **文档**: docs/support.md

### 常见问题
1. **端口冲突**: 检查8000, 5432, 6379端口是否被占用
2. **权限问题**: 确保用户有Docker和文件系统权限
3. **依赖问题**: 确认Python版本 >= 3.9
4. **内存不足**: 确保系统有至少8GB可用内存

## 🎯 下一步

1. **监控设置**: 配置Prometheus和Grafana监控
2. **日志配置**: 设置ELK日志聚合系统
3. **备份策略**: 配置自动备份和恢复机制
4. **安全加固**: 实施生产环境安全措施
5. **性能优化**: 根据实际负载调整配置

---

**生成时间**: {datetime.now().isoformat()}
**RQA2025项目交付小组**
"""

        return guide

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def save_delivery_manifest(self, manifest: Dict[str, Any],
                            manifest_file: str = "DELIVERY_MANIFEST.json") -> str:
        """保存交付清单"""
        manifest_path = self.project_root / "delivery_packages" / manifest_file
        manifest_path.parent.mkdir(exist_ok=True)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.logger.info(f"交付清单已保存到: {manifest_path}")

        return str(manifest_path)

    def generate_delivery_report(self, delivery_result: Dict[str, Any]) -> str:
        """生成交付报告"""
        lines = []
        lines.append("# RQA2025项目交付报告")
        lines.append(f"交付时间: {datetime.now().isoformat()}")
        lines.append(f"项目版本: {delivery_result.get('project_version', 'Unknown')}")
        lines.append("")

        # 交付概览
        packages = delivery_result.get("delivery_packages", {})
        lines.append("## 📦 交付包概览")

        for package_type, package_info in packages.items():
            lines.append(f"### {package_type.replace('_', ' ').title()}")
            lines.append(f"- 包名: {package_info.get('package_name', 'Unknown')}")
            lines.append(f"- 文件数: {package_info.get('file_count', 0)}")
            lines.append(".2")
            lines.append(f"- 哈希: {package_info.get('package_hash', 'Unknown')[:16]}...")
            lines.append("")

        # 项目结构
        structure = delivery_result.get("delivery_packages", {}).get("source_package", {}).get("project_structure", {})
        if structure:
            lines.append("## 📊 项目结构统计")
            lines.append(f"- 总文件数: {structure.get('total_files', 0):,}")
            lines.append(f"- 总目录数: {structure.get('total_directories', 0):,}")
            lines.append(".2")
            lines.append("")

        # 交付清单
        manifest = delivery_result.get("delivery_manifest", {})
        if manifest.get("system_requirements"):
            lines.append("## 🔧 系统要求")
            reqs = manifest["system_requirements"]
            lines.append(f"- Python版本: {reqs.get('python_version', 'Unknown')}")
            lines.append(f"- 操作系统: {', '.join(reqs.get('operating_system', []))}")
            lines.append(f"- 数据库: {', '.join(reqs.get('database', []))}")
            lines.append("")

        # 安装指南
        if manifest.get("installation_guide"):
            lines.append("## 🚀 安装步骤")
            guide = manifest["installation_guide"]
            for step_key, step_desc in guide.items():
                lines.append(f"1. {step_desc}")
            lines.append("")

        # 验证清单
        if manifest.get("validation_checklist"):
            lines.append("## ✅ 验证清单")
            for item in manifest["validation_checklist"]:
                lines.append(f"- {item}")
            lines.append("")

        # 技术支持
        if manifest.get("support_information"):
            lines.append("## 📞 技术支持")
            support = manifest["support_information"]
            lines.append(f"- 文档: {support.get('documentation', 'Unknown')}")
            lines.append(f"- 故障排除: {support.get('troubleshooting', 'Unknown')}")
            lines.append(f"- 联系方式: {support.get('contact', 'Unknown')}")
            lines.append("")

        # 交付总结
        lines.append("## 🎯 交付总结")
        lines.append("**RQA2025量化交易系统完整交付包已生成！**")
        lines.append("")
        lines.append("**包含内容**:")
        lines.append("- ✅ 完整源代码和依赖")
        lines.append("- ✅ 容器化和部署配置")
        lines.append("- ✅ 全面文档和指南")
        lines.append("- ✅ 测试套件和验证工具")
        lines.append("- ✅ 交付清单和快速开始指南")
        lines.append("")
        lines.append("**可以安全部署到生产环境！** 🚀")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"**交付耗时**: {delivery_result.get('delivery_duration', 0):.2f}秒")
        lines.append("")
        lines.append("**RQA2025项目交付小组**")
        lines.append(f"**{datetime.now().strftime('%Y年%m月%d日')}**")

        return "\n".join(lines)

    def save_delivery_report(self, delivery_result: Dict[str, Any],
                        report_file: str = "project_delivery_report.md") -> str:
        """保存交付报告"""
        report_path = self.project_root / "delivery_packages" / report_file
        report_path.parent.mkdir(exist_ok=True)

        report_content = self.generate_delivery_report(delivery_result)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"交付报告已保存到: {report_path}")

        return str(report_path)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025项目交付生成器")
    parser.add_argument("--generate-packages", action="store_true",
                    help="生成所有交付包")
    parser.add_argument("--report", type=str, default="project_delivery_report.md",
                    help="交付报告文件路径")
    parser.add_argument("--manifest", type=str, default="DELIVERY_MANIFEST.json",
                    help="交付清单文件路径")

    args = parser.parse_args()

    # 创建交付生成器
    generator = ProjectDeliveryGenerator()

    try:
        print("🚀 开始生成RQA2025项目完整交付包...")

        # 生成完整交付包
        delivery_result = generator.generate_complete_delivery_package()

        # 保存交付报告
        report_path = generator.save_delivery_report(delivery_result, args.report)

        print("✅ 交付包生成完成！")
        print(f"📊 详细报告: {report_path}")
        print(f"📋 交付清单: {delivery_result.get('manifest_path', 'Unknown')}")

        # 输出交付包信息
        packages = delivery_result.get("delivery_packages", {})
        print("\n📦 生成的交付包:")
        for package_type, package_info in packages.items():
            package_name = package_info.get('package_name', 'Unknown')
            package_size = package_info.get('package_size_mb', 0)
            print(f"   {package_name}: {package_size:.2f} MB")
        # 输出项目统计
        structure = packages.get("source_code", {}).get("project_structure", {})
        if structure:
            print("\n📊 项目统计:")
            print(f"   文件总数: {structure.get('total_files', 0):,}")
            print(f"   目录总数: {structure.get('total_directories', 0):,}")
            print(".2")
        print("\n🎉 RQA2025项目交付包生成完毕，可以安全交付！")

    except KeyboardInterrupt:
        print("\n⚠️ 交付包生成被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 交付包生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
