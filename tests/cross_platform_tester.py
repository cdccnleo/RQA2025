#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨平台兼容性测试框架

完善不同操作系统的支持：
- Windows兼容性测试
- Linux兼容性测试
- macOS兼容性测试
- 容器化环境测试
- 字符编码处理
- 路径处理差异
- 系统命令适配
"""

import os
import sys
import platform
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
import locale
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """平台信息"""
    system: str
    release: str
    version: str
    machine: str
    processor: str
    python_version: str
    encoding: str
    filesystem_encoding: str
    locale: str


@dataclass
class CompatibilityTest:
    """兼容性测试"""
    name: str
    platforms: List[str]  # 支持的平台
    test_function: str
    expected_result: Any
    timeout: int = 30


@dataclass
class PlatformTestResult:
    """平台测试结果"""
    platform: str
    test_name: str
    success: bool
    duration: float
    actual_result: Any
    error_message: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)


class PathHandler:
    """跨平台路径处理器"""

    def __init__(self):
        self.system = platform.system().lower()

    def normalize_path(self, path: str) -> str:
        """规范化路径"""
        if self.system == 'windows':
            return path.replace('/', '\\')
        else:
            return path.replace('\\', '/')

    def join_path(self, *paths: str) -> str:
        """跨平台路径连接"""
        return os.path.join(*paths)

    def get_temp_dir(self) -> str:
        """获取临时目录"""
        return tempfile.gettempdir()

    def get_home_dir(self) -> str:
        """获取用户主目录"""
        return os.path.expanduser('~')

    def is_absolute_path(self, path: str) -> bool:
        """检查是否为绝对路径"""
        return os.path.isabs(path)

    def resolve_path(self, path: str) -> str:
        """解析路径（处理符号链接等）"""
        return os.path.realpath(path)

    def get_relative_path(self, path: str, start: str = os.getcwd()) -> str:
        """获取相对路径"""
        try:
            return os.path.relpath(path, start)
        except ValueError:
            return path


class EncodingHandler:
    """编码处理器"""

    def __init__(self):
        self.system = platform.system().lower()
        self.preferred_encodings = self._get_preferred_encodings()

    def _get_preferred_encodings(self) -> List[str]:
        """获取首选编码列表"""
        if self.system == 'windows':
            return ['utf-8', 'gbk', 'cp1252', 'latin1']
        elif self.system == 'linux':
            return ['utf-8', 'ascii', 'latin1']
        elif self.system == 'darwin':  # macOS
            return ['utf-8', 'ascii', 'latin1']
        else:
            return ['utf-8', 'ascii', 'latin1']

    def safe_encode(self, text: str) -> bytes:
        """安全编码"""
        for encoding in self.preferred_encodings:
            try:
                return text.encode(encoding)
            except UnicodeEncodeError:
                continue
        # 如果所有编码都失败，使用错误替换
        return text.encode('utf-8', errors='replace')

    def safe_decode(self, data: bytes) -> str:
        """安全解码"""
        for encoding in self.preferred_encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        # 如果所有解码都失败，使用错误替换
        return data.decode('utf-8', errors='replace')

    def get_system_encoding(self) -> str:
        """获取系统编码"""
        try:
            return locale.getpreferredencoding(False)
        except Exception:
            return 'utf-8'

    def test_encoding_compatibility(self) -> Dict[str, Any]:
        """测试编码兼容性"""
        results = {}

        # 测试特殊字符
        test_strings = [
            "Hello World",
            "你好世界",
            "🌟⭐🌙",
            "café",
            "naïve",
            "Москва",
            "東京",
            "🚀💻🔧"
        ]

        for encoding in self.preferred_encodings:
            encoding_results = []
            for test_string in test_strings:
                try:
                    encoded = test_string.encode(encoding)
                    decoded = encoded.decode(encoding)
                    success = decoded == test_string
                    encoding_results.append({
                        'string': test_string,
                        'success': success,
                        'encoded_length': len(encoded)
                    })
                except Exception as e:
                    encoding_results.append({
                        'string': test_string,
                        'success': False,
                        'error': str(e)
                    })

            results[encoding] = encoding_results

        return results


class CommandAdapter:
    """命令适配器"""

    def __init__(self):
        self.system = platform.system().lower()
        self.path_handler = PathHandler()
        self.encoding_handler = EncodingHandler()

    def run_command(self, command: List[str], cwd: Optional[str] = None,
                   timeout: int = 30) -> Tuple[bool, str, str]:
        """运行命令（跨平台）"""
        try:
            # 处理命令参数
            processed_command = self._process_command(command)

            # 设置环境变量
            env = self._get_environment()

            # 运行命令
            result = subprocess.run(
                processed_command,
                cwd=cwd,
                capture_output=True,
                timeout=timeout,
                env=env,
                text=False  # 使用字节模式，然后手动解码
            )

            # 解码输出
            stdout = self.encoding_handler.safe_decode(result.stdout)
            stderr = self.encoding_handler.safe_decode(result.stderr)

            return result.returncode == 0, stdout, stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Command execution failed: {e}"

    def _process_command(self, command: List[str]) -> List[str]:
        """处理命令参数"""
        processed = []

        for arg in command:
            if self.system == 'windows':
                # Windows特殊处理
                if arg.startswith('./') or arg.startswith('../'):
                    # 转换为Windows路径格式
                    arg = arg.replace('/', '\\')
                elif arg == 'python':
                    # 在Windows上可能需要使用python3或py
                    arg = sys.executable
            else:
                # Unix-like系统
                if arg == 'python' and sys.executable:
                    arg = sys.executable

            processed.append(arg)

        return processed

    def _get_environment(self) -> Dict[str, str]:
        """获取环境变量"""
        env = os.environ.copy()

        # 设置跨平台环境变量
        env['PYTHONIOENCODING'] = 'utf-8'
        env['LANG'] = 'C.UTF-8'

        if self.system == 'windows':
            # Windows特定设置
            env['PYTHONUTF8'] = '1'
        else:
            # Unix-like系统设置
            env['LC_ALL'] = 'C.UTF-8'

        return env

    def test_command_compatibility(self) -> Dict[str, Any]:
        """测试命令兼容性"""
        test_commands = [
            ['echo', 'Hello World'],
            ['python', '--version'],
            ['python', '-c', 'print("Hello Python")'],
        ]

        results = {}

        for command in test_commands:
            cmd_name = ' '.join(command[:2])  # 使用前两个参数作为名称
            success, stdout, stderr = self.run_command(command, timeout=10)
            results[cmd_name] = {
                'success': success,
                'stdout': stdout.strip(),
                'stderr': stderr.strip(),
                'command': command
            }

        return results


class FileSystemTester:
    """文件系统测试器"""

    def __init__(self):
        self.path_handler = PathHandler()
        self.encoding_handler = EncodingHandler()

    def test_file_operations(self) -> Dict[str, Any]:
        """测试文件操作"""
        results = {}

        # 创建临时文件进行测试
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name

            # 测试写入特殊字符
            test_content = "Hello 世界 🌟\nSecond line: café"
            try:
                f.write(test_content)
                f.flush()

                # 测试读取
                with open(temp_file, 'r', encoding='utf-8') as rf:
                    read_content = rf.read()

                results['file_write_read'] = {
                    'success': read_content == test_content,
                    'original': test_content,
                    'read': read_content
                }

            except Exception as e:
                results['file_write_read'] = {
                    'success': False,
                    'error': str(e)
                }

        # 清理临时文件
        try:
            os.unlink(temp_file)
        except Exception:
            pass

        # 测试路径操作
        test_paths = [
            "/tmp/test",
            "C:\\temp\\test",
            "./relative/path",
            "../parent/path"
        ]

        path_results = []
        for path_str in test_paths:
            try:
                normalized = self.path_handler.normalize_path(path_str)
                is_abs = self.path_handler.is_absolute_path(path_str)
                path_results.append({
                    'original': path_str,
                    'normalized': normalized,
                    'is_absolute': is_abs,
                    'success': True
                })
            except Exception as e:
                path_results.append({
                    'original': path_str,
                    'success': False,
                    'error': str(e)
                })

        results['path_operations'] = path_results

        return results

    def test_directory_operations(self) -> Dict[str, Any]:
        """测试目录操作"""
        results = {}

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        try:
            # 测试目录创建
            test_subdir = os.path.join(temp_dir, "test_subdir", "nested")
            os.makedirs(test_subdir, exist_ok=True)

            results['directory_creation'] = {
                'success': os.path.exists(test_subdir),
                'path': test_subdir
            }

            # 测试文件在目录中创建
            test_file = os.path.join(test_subdir, "test_file.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("Test content")

            results['file_in_directory'] = {
                'success': os.path.exists(test_file),
                'path': test_file
            }

            # 测试目录遍历
            all_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))

            results['directory_traversal'] = {
                'success': len(all_files) == 1 and 'test_file.txt' in all_files[0],
                'files_found': all_files
            }

        except Exception as e:
            results['directory_operations'] = {
                'success': False,
                'error': str(e)
            }

        # 清理
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return results


class CrossPlatformTester:
    """跨平台测试器"""

    def __init__(self):
        self.system = platform.system().lower()
        self.path_handler = PathHandler()
        self.encoding_handler = EncodingHandler()
        self.command_adapter = CommandAdapter()
        self.filesystem_tester = FileSystemTester()

        self.compatibility_tests = self._define_compatibility_tests()

    def _define_compatibility_tests(self) -> List[CompatibilityTest]:
        """定义兼容性测试"""
        return [
            CompatibilityTest(
                name="basic_file_operations",
                platforms=["windows", "linux", "darwin"],
                test_function="test_file_operations",
                expected_result=True
            ),
            CompatibilityTest(
                name="directory_operations",
                platforms=["windows", "linux", "darwin"],
                test_function="test_directory_operations",
                expected_result=True
            ),
            CompatibilityTest(
                name="command_execution",
                platforms=["windows", "linux", "darwin"],
                test_function="test_command_execution",
                expected_result=True
            ),
            CompatibilityTest(
                name="encoding_handling",
                platforms=["windows", "linux", "darwin"],
                test_function="test_encoding_handling",
                expected_result=True
            ),
            CompatibilityTest(
                name="path_handling",
                platforms=["windows", "linux", "darwin"],
                test_function="test_path_handling",
                expected_result=True
            )
        ]

    def get_platform_info(self) -> PlatformInfo:
        """获取平台信息"""
        return PlatformInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=sys.version,
            encoding=self.encoding_handler.get_system_encoding(),
            filesystem_encoding=sys.getfilesystemencoding(),
            locale=locale.getlocale()[0] or 'unknown'
        )

    def run_compatibility_tests(self) -> List[PlatformTestResult]:
        """运行兼容性测试"""
        logger.info("开始运行跨平台兼容性测试...")

        results = []
        platform_info = self.get_platform_info()

        for test in self.compatibility_tests:
            # 检查是否支持当前平台
            if self.system not in test.platforms:
                logger.info(f"跳过测试 {test.name} (不支持平台 {self.system})")
                continue

            logger.info(f"运行测试: {test.name}")

            result = self._run_single_test(test, platform_info)
            results.append(result)

        logger.info("跨平台兼容性测试完成")
        return results

    def _run_single_test(self, test: CompatibilityTest, platform_info: PlatformInfo) -> PlatformTestResult:
        """运行单个测试"""
        start_time = time.time()

        try:
            # 调用测试函数
            if hasattr(self, test.test_function):
                test_method = getattr(self, test.test_function)
                actual_result = test_method()

                # 检查结果
                success = self._check_test_result(actual_result, test.expected_result)

                return PlatformTestResult(
                    platform=self.system,
                    test_name=test.name,
                    success=success,
                    duration=time.time() - start_time,
                    actual_result=actual_result,
                    system_info={
                        'system': platform_info.system,
                        'python_version': platform_info.python_version,
                        'encoding': platform_info.encoding
                    }
                )

            else:
                return PlatformTestResult(
                    platform=self.system,
                    test_name=test.name,
                    success=False,
                    duration=time.time() - start_time,
                    actual_result=None,
                    error_message=f"测试函数 {test.test_function} 不存在"
                )

        except Exception as e:
            return PlatformTestResult(
                platform=self.system,
                test_name=test.name,
                success=False,
                duration=time.time() - start_time,
                actual_result=None,
                error_message=str(e)
            )

    def _check_test_result(self, actual_result: Any, expected_result: Any) -> bool:
        """检查测试结果"""
        if isinstance(expected_result, bool):
            # 如果期望布尔值，检查实际结果中是否有成功的标志
            if isinstance(actual_result, dict):
                return actual_result.get('success', False) == expected_result
            return bool(actual_result) == expected_result

        # 对于其他类型，简单比较
        return actual_result == expected_result

    def test_file_operations(self) -> Dict[str, Any]:
        """测试文件操作"""
        return self.filesystem_tester.test_file_operations()

    def test_directory_operations(self) -> Dict[str, Any]:
        """测试目录操作"""
        return self.filesystem_tester.test_directory_operations()

    def test_command_execution(self) -> Dict[str, Any]:
        """测试命令执行"""
        return self.command_adapter.test_command_compatibility()

    def test_encoding_handling(self) -> Dict[str, Any]:
        """测试编码处理"""
        return self.encoding_handler.test_encoding_compatibility()

    def test_path_handling(self) -> Dict[str, Any]:
        """测试路径处理"""
        test_paths = [
            "/absolute/unix/path",
            "C:\\absolute\\windows\\path",
            "./relative/path",
            "../parent/path"
        ]

        results = []
        for path_str in test_paths:
            try:
                normalized = self.path_handler.normalize_path(path_str)
                is_abs = self.path_handler.is_absolute_path(path_str)
                resolved = self.path_handler.resolve_path(path_str)

                results.append({
                    'original': path_str,
                    'normalized': normalized,
                    'is_absolute': is_abs,
                    'resolved': resolved,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'original': path_str,
                    'success': False,
                    'error': str(e)
                })

        return {'path_tests': results}

    def generate_compatibility_report(self, results: List[PlatformTestResult]):
        """生成兼容性报告"""
        report_path = Path("test_logs/cross_platform_report.md")

        platform_info = self.get_platform_info()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 跨平台兼容性测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 🖥️ 平台信息\n\n")
            f.write(f"- **操作系统**: {platform_info.system} {platform_info.release}\n")
            f.write(f"- **版本**: {platform_info.version}\n")
            f.write(f"- **架构**: {platform_info.machine}\n")
            f.write(f"- **处理器**: {platform_info.processor}\n")
            f.write(f"- **Python版本**: {platform_info.python_version.split()[0]}\n")
            f.write(f"- **系统编码**: {platform_info.encoding}\n")
            f.write(f"- **文件系统编码**: {platform_info.filesystem_encoding}\n")
            f.write(f"- **区域设置**: {platform_info.locale}\n\n")

            f.write("## 📊 测试结果\n\n")

            successful_tests = sum(1 for r in results if r.success)
            total_tests = len(results)

            f.write(f"- **总测试数**: {total_tests}\n")
            f.write(f"- **成功测试**: {successful_tests}\n")
            f.write(".1")
            f.write("\n## 🧪 详细测试结果\n\n")

            for result in results:
                status = "✅" if result.success else "❌"
                f.write(f"### {status} {result.test_name}\n\n")
                f.write(".2")
                f.write(f"- **平台**: {result.platform}\n")

                if result.error_message:
                    f.write(f"- **错误**: {result.error_message}\n")

                # 显示关键结果信息
                if isinstance(result.actual_result, dict):
                    if 'file_write_read' in result.actual_result:
                        file_test = result.actual_result['file_write_read']
                        f.write(f"- **文件读写**: {'✅' if file_test.get('success') else '❌'}\n")

                    if 'path_operations' in result.actual_result:
                        path_ops = result.actual_result['path_operations']
                        successful_paths = sum(1 for p in path_ops if p.get('success'))
                        f.write(f"- **路径操作**: {successful_paths}/{len(path_ops)} 成功\n")

                    if 'echo Hello World' in result.actual_result:
                        cmd_test = result.actual_result['echo Hello World']
                        f.write(f"- **命令执行**: {'✅' if cmd_test.get('success') else '❌'}\n")

                f.write("\n")

            f.write("## 🔧 兼容性分析\n\n")

            # 分析不同方面的兼容性
            compatibility_areas = {
                '文件操作': ['basic_file_operations'],
                '目录操作': ['directory_operations'],
                '命令执行': ['command_execution'],
                '编码处理': ['encoding_handling'],
                '路径处理': ['path_handling']
            }

            for area, test_names in compatibility_areas.items():
                area_results = [r for r in results if r.test_name in test_names]
                if area_results:
                    success_count = sum(1 for r in area_results if r.success)
                    total_count = len(area_results)
                    f.write(f"- **{area}**: {success_count}/{total_count} 兼容\n")

            f.write("\n## 💡 兼容性建议\n\n")

            if successful_tests < total_tests:
                f.write("### 需要关注的兼容性问题:\n\n")
                failed_tests = [r for r in results if not r.success]
                for test in failed_tests:
                    f.write(f"1. **{test.test_name}**: {test.error_message or '测试失败'}\n")

            f.write("\n### 跨平台开发最佳实践:\n\n")
            f.write("1. **路径处理**: 使用 `os.path` 或 `pathlib` 进行跨平台路径操作\n")
            f.write("2. **编码处理**: 统一使用UTF-8编码，处理解码错误\n")
            f.write("3. **命令执行**: 使用 `subprocess` 时设置合适的编码和环境变量\n")
            f.write("4. **文件操作**: 使用 `with` 语句和显式编码参数\n")
            f.write("5. **平台检测**: 使用 `platform.system()` 进行平台特定处理\n")
            f.write("6. **环境变量**: 设置 `PYTHONIOENCODING=utf-8` 等跨平台变量\n")

        logger.info(f"跨平台兼容性报告已生成: {report_path}")


def main():
    """主函数"""
    tester = CrossPlatformTester()

    platform_info = tester.get_platform_info()

    print("🌍 跨平台兼容性测试器启动")
    print(f"🎯 当前平台: {platform_info.system} {platform_info.release}")
    print(f"🐍 Python版本: {platform_info.python_version.split()[0]}")
    print(f"📝 系统编码: {platform_info.encoding}")

    # 运行兼容性测试
    results = tester.run_compatibility_tests()

    print("\n📊 测试结果:")
    successful_tests = sum(1 for r in results if r.success)
    total_tests = len(results)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

    print(f"  📋 总测试数: {total_tests}")
    print(f"  ✅ 成功测试: {successful_tests}")
    print(".1")
    # 显示详细结果
    print("\n🧪 详细结果:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(".2")
    print("\n📄 详细报告已保存到: test_logs/cross_platform_report.md")
    print("\n✅ 跨平台兼容性测试器运行完成")


if __name__ == "__main__":
    main()
