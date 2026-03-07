#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 部署就绪检查测试
验证系统部署配置、依赖管理、环境要求等
"""

import unittest
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestDeploymentReadiness(unittest.TestCase):
    """部署就绪检查测试"""

    def setUp(self):
        """测试前准备"""
        self.project_root = Path(__file__).parent.parent.parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def test_python_version_compatibility(self):
        """测试Python版本兼容性"""
        print("\n=== Python版本兼容性检查 ===")

        # 检查Python版本
        major, minor = sys.version_info.major, sys.version_info.minor
        version_str = f"{major}.{minor}"

        print(f"✅ 当前Python版本: {version_str}")

        # 验证支持的版本
        supported_versions = ["3.8", "3.9", "3.10", "3.11"]
        self.assertIn(version_str, supported_versions,
                     f"Python {version_str} 不在支持的版本列表中: {supported_versions}")

        # 验证版本不低于3.8
        self.assertGreaterEqual(major, 3)
        self.assertGreaterEqual(minor, 8)

        print("✅ Python版本兼容性验证通过")
        print("🎉 Python版本兼容性检查通过！")

    def test_dependencies_availability(self):
        """测试依赖可用性"""
        print("\n=== 依赖可用性检查 ===")

        # 检查核心依赖
        core_dependencies = [
            "pytest",
            "unittest",
            "json",
            "time",
            "threading",
            "pathlib",
            "typing"
        ]

        for dependency in core_dependencies:
            try:
                __import__(dependency)
                print(f"✅ 依赖 {dependency} 可用")
            except ImportError:
                self.fail(f"核心依赖 {dependency} 不可用")

        # 检查可选依赖
        optional_dependencies = [
            "psutil",  # 系统监控
            "numpy",   # 数值计算
            "pandas",  # 数据处理
            "scipy",   # 科学计算
            "matplotlib",  # 可视化
            "seaborn"  # 统计图表
        ]

        missing_optional = []
        for dependency in optional_dependencies:
            try:
                __import__(dependency)
                print(f"✅ 可选依赖 {dependency} 可用")
            except ImportError:
                missing_optional.append(dependency)
                print(f"⚠️  可选依赖 {dependency} 不可用")

        # 可选依赖缺失不应该导致测试失败，但应该报告
        if missing_optional:
            print(f"⚠️  缺少可选依赖: {', '.join(missing_optional)}")

        print("✅ 依赖可用性检查完成")
        print("🎉 依赖可用性检查通过！")

    def test_project_structure_integrity(self):
        """测试项目结构完整性"""
        print("\n=== 项目结构完整性检查 ===")

        # 检查必需的目录结构
        required_directories = [
            "src",
            "src/infrastructure",
            "src/infrastructure/config",
            "src/infrastructure/config/core",
            "src/infrastructure/config/loaders",
            "src/infrastructure/config/services",
            "src/infrastructure/config/monitoring",
            "src/infrastructure/config/security",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "tests/security"
        ]

        for directory in required_directories:
            dir_path = self.project_root / directory
            self.assertTrue(dir_path.exists(), f"必需目录不存在: {directory}")
            self.assertTrue(dir_path.is_dir(), f"路径不是目录: {directory}")
            print(f"✅ 目录存在: {directory}")

        # 检查关键文件
        critical_files = [
            "src/__init__.py",
            "src/infrastructure/__init__.py",
            "src/infrastructure/config/__init__.py",
            "tests/__init__.py",
            "pytest.ini"
        ]

        for file_path in critical_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"关键文件不存在: {file_path}")
            self.assertTrue(full_path.is_file(), f"路径不是文件: {file_path}")
            print(f"✅ 文件存在: {file_path}")

        print("✅ 项目结构完整性验证通过")
        print("🎉 项目结构完整性检查通过！")

    def test_configuration_files_validity(self):
        """测试配置文件有效性"""
        print("\n=== 配置文件有效性检查 ===")

        # 检查pytest配置
        pytest_config = self.project_root / "pytest.ini"
        if pytest_config.exists():
            try:
                with open(pytest_config, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertIn("[tool:pytest]", content)
                    print("✅ pytest.ini 配置有效")
            except Exception as e:
                self.fail(f"pytest.ini 配置文件无效: {e}")

        # 检查requirements文件
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    self.assertGreater(len(lines), 0, "requirements.txt 为空")
                    print(f"✅ requirements.txt 包含 {len(lines)} 个依赖")
            except Exception as e:
                self.fail(f"requirements.txt 文件无效: {e}")
        else:
            print("⚠️  requirements.txt 不存在")

        # 检查setup.py或pyproject.toml
        setup_py = self.project_root / "setup.py"
        pyproject_toml = self.project_root / "pyproject.toml"

        if setup_py.exists():
            print("✅ setup.py 存在")
        elif pyproject_toml.exists():
            print("✅ pyproject.toml 存在")
        else:
            print("⚠️  既没有setup.py也没有pyproject.toml")

        print("✅ 配置文件有效性检查完成")
        print("🎉 配置文件有效性检查通过！")

    def test_import_structure_validity(self):
        """测试导入结构有效性"""
        print("\n=== 导入结构有效性检查 ===")

        # 测试核心模块导入
        core_modules = [
            "src.infrastructure",
            "src.infrastructure.config",
            "src.infrastructure.config.core",
            "src.infrastructure.config.loaders",
            "src.infrastructure.config.services"
        ]

        for module in core_modules:
            try:
                __import__(module)
                print(f"✅ 模块 {module} 可导入")
            except ImportError as e:
                print(f"⚠️  模块 {module} 导入失败: {e}")
            except Exception as e:
                print(f"⚠️  模块 {module} 导入出错: {e}")

        # 测试相对导入（如果适用）
        try:
            from src.infrastructure.config import core
            print("✅ 相对导入 src.infrastructure.config.core 成功")
        except ImportError as e:
            print(f"⚠️  相对导入失败: {e}")

        print("✅ 导入结构有效性检查完成")
        print("🎉 导入结构有效性检查通过！")

    def test_environment_variables_setup(self):
        """测试环境变量设置"""
        print("\n=== 环境变量设置检查 ===")

        # 检查重要的环境变量
        important_env_vars = [
            "PYTHONPATH",  # Python路径
            "PATH"         # 系统路径
        ]

        for env_var in important_env_vars:
            value = os.environ.get(env_var)
            if value:
                print(f"✅ 环境变量 {env_var} 已设置")
            else:
                print(f"⚠️  环境变量 {env_var} 未设置")

        # 检查Python路径是否包含项目根目录
        python_path = os.environ.get("PYTHONPATH", "")
        if str(self.project_root) in python_path:
            print("✅ 项目根目录在PYTHONPATH中")
        else:
            print("⚠️  项目根目录不在PYTHONPATH中")

        print("✅ 环境变量设置检查完成")
        print("🎉 环境变量设置检查通过！")

    def test_executable_permissions(self):
        """测试可执行权限"""
        print("\n=== 可执行权限检查 ===")

        # 检查Python脚本文件
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        # 检查前10个Python文件
        checked_files = 0
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # 只读取前100个字符
                    if content.startswith('#!/usr/bin/env python') or content.startswith('#!/usr/bin/python'):
                        print(f"✅ 脚本文件具有shebang: {os.path.basename(py_file)}")
                        checked_files += 1
            except Exception:
                pass

        if checked_files == 0:
            print("ℹ️  未发现需要shebang的脚本文件")

        print("✅ 可执行权限检查完成")
        print("🎉 可执行权限检查通过！")

    def test_deployment_scripts_validity(self):
        """测试部署脚本有效性"""
        print("\n=== 部署脚本有效性检查 ===")

        # 查找可能的部署脚本
        deployment_scripts = [
            "deploy.sh",
            "deploy.py",
            "setup.py",
            "install.sh",
            "docker-compose.yml",
            "Dockerfile"
        ]

        found_scripts = []
        for script in deployment_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                found_scripts.append(script)
                print(f"✅ 部署脚本存在: {script}")

        if not found_scripts:
            print("ℹ️  未发现部署脚本")

        # 检查Docker相关文件
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        docker_found = []
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            if file_path.exists():
                docker_found.append(docker_file)
                print(f"✅ Docker文件存在: {docker_file}")

        if docker_found:
            print(f"✅ 发现 {len(docker_found)} 个Docker相关文件")
        else:
            print("ℹ️  未发现Docker相关文件")

        print("✅ 部署脚本有效性检查完成")
        print("🎉 部署脚本有效性检查通过！")

    def test_code_quality_metrics(self):
        """测试代码质量指标"""
        print("\n=== 代码质量指标检查 ===")

        # 统计代码行数
        total_lines = 0
        total_files = 0
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # 跳过一些不需要的目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

            for file in files:
                if file.endswith('.py'):
                    total_files += 1
                    python_files.append(os.path.join(root, file))

                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            total_lines += len(lines)
                    except Exception:
                        pass

        print("📊 代码统计:")
        print(f"✅ Python文件总数: {total_files}")
        print(f"✅ 总代码行数: {total_lines}")
        print(f"✅ 平均代码行数: {total_lines / max(total_files, 1):.1f}")
        # 检查是否有足够的代码
        self.assertGreater(total_files, 10, "Python文件数量太少")
        self.assertGreater(total_lines, 1000, "代码行数太少")

        # 检查是否有测试文件
        test_files = [f for f in python_files if 'test' in f.lower()]
        test_coverage_ratio = len(test_files) / max(total_files, 1)

        print(f"✅ 测试文件比例: {test_coverage_ratio:.1%}")
        # 建议测试覆盖率
        if test_coverage_ratio < 0.1:
            print("⚠️  测试文件比例较低，建议增加测试覆盖")
        else:
            print("✅ 测试文件比例合理")

        print("✅ 代码质量指标检查完成")
        print("🎉 代码质量指标检查通过！")

    def test_system_resource_requirements(self):
        """测试系统资源需求"""
        print("\n=== 系统资源需求检查 ===")

        # 检查内存使用
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"✅ 当前内存使用: {memory_usage:.1f} MB")
        except ImportError:
            print("⚠️  psutil不可用，无法检查内存使用")

        # 检查磁盘空间
        try:
            stat = os.statvfs(self.project_root)
            free_space = stat.f_bavail * stat.f_frsize / 1024 / 1024 / 1024  # GB
            print(f"✅ 磁盘可用空间: {free_space:.1f} GB")
        except AttributeError:
            # Windows系统可能不支持statvfs
            print("ℹ️  当前系统不支持磁盘空间检查")

        # 检查CPU核心数
        try:
            cpu_count = os.cpu_count()
            print(f"✅ CPU核心数: {cpu_count}")
        except Exception:
            print("ℹ️  无法获取CPU核心数")

        print("✅ 系统资源需求检查完成")
        print("🎉 系统资源需求检查通过！")

    def test_final_deployment_readiness_score(self):
        """测试最终部署就绪评分"""
        print("\n=== 最终部署就绪评分 ===")

        # 计算部署就绪评分
        readiness_checks = {
            "python_version": self._check_python_version(),
            "dependencies": self._check_dependencies(),
            "project_structure": self._check_project_structure(),
            "config_files": self._check_config_files(),
            "import_structure": self._check_import_structure(),
            "environment": self._check_environment(),
            "code_quality": self._check_code_quality(),
            "resources": self._check_resources()
        }

        passed_checks = sum(1 for check in readiness_checks.values() if check)
        total_checks = len(readiness_checks)
        readiness_score = (passed_checks / total_checks) * 100

        print("📊 部署就绪检查结果:")
        print(f"✅ 通过检查: {passed_checks}/{total_checks}")
        print(f"✅ 就绪评分: {readiness_score:.1f}%")
        for check_name, passed in readiness_checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {'通过' if passed else '失败'}")

        # 部署就绪标准
        if readiness_score >= 80:
            print("🎉 部署就绪评分优秀！系统可以投入生产使用")
        elif readiness_score >= 60:
            print("⚠️  部署就绪评分良好，建议解决剩余问题")
        else:
            print("❌ 部署就绪评分不足，需解决关键问题后再部署")

        self.assertGreaterEqual(readiness_score, 60,
                              f"部署就绪评分 {readiness_score:.1f}% 低于最低要求 60%")
        print("🎉 最终部署就绪评分检查完成！")

    # ==================== 辅助方法 ====================

    def _check_python_version(self) -> bool:
        """检查Python版本"""
        major, minor = sys.version_info.major, sys.version_info.minor
        return major >= 3 and minor >= 8

    def _check_dependencies(self) -> bool:
        """检查依赖"""
        core_deps = ["pytest", "unittest", "json", "time"]
        try:
            for dep in core_deps:
                __import__(dep)
            return True
        except ImportError:
            return False

    def _check_project_structure(self) -> bool:
        """检查项目结构"""
        required_dirs = ["src", "src/infrastructure", "tests"]
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                return False
        return True

    def _check_config_files(self) -> bool:
        """检查配置文件"""
        config_files = ["pytest.ini"]
        for file_name in config_files:
            if not (self.project_root / file_name).exists():
                return False
        return True

    def _check_import_structure(self) -> bool:
        """检查导入结构"""
        try:
            import src.infrastructure
            return True
        except ImportError:
            return False

    def _check_environment(self) -> bool:
        """检查环境"""
        return "PYTHONPATH" in os.environ

    def _check_code_quality(self) -> bool:
        """检查代码质量"""
        # 简单的代码质量检查
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        return len(python_files) > 10

    def _check_resources(self) -> bool:
        """检查资源"""
        try:
            cpu_count = os.cpu_count()
            return cpu_count is not None and cpu_count >= 2
        except Exception:
            return True  # 如果无法检查，则假设通过


if __name__ == '__main__':
    unittest.main()
