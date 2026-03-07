#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平台兼容性修复脚本
修复硬编码路径、平台特定代码和导入路径问题
"""

import re
from pathlib import Path
from typing import List, Dict


class PlatformCompatibilityFixer:
    """平台兼容性修复器"""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.fixes_applied = []

    def fix_hardcoded_paths(self) -> List[str]:
        """修复硬编码路径问题"""
        print("🔧 修复硬编码路径问题...")

        # 查找所有需要修复的文件
        files_to_fix = [
            "tests/integration/api/test_feature_api.py",
            "tests/unit/core/test_api_gateway.py",
            "tests/unit/core/test_service_container.py",
            "tests/unit/data/test_data_integration.py",
            "tests/unit/data/test_financial_loader.py",
            "tests/unit/data/test_forex_loader.py",
            "tests/unit/data/test_macro_loader.py",
            "tests/unit/data/test_news_loader.py",
            "tests/unit/infrastructure/config/test_config_system.py",
            "tests/unit/infrastructure/logging/test_logging_system.py",
            "tests/unit/infrastructure/monitoring/test_performance_benchmark.py",
            "tests/unit/infrastructure/utils/test_utils.py"
        ]

        fixed_files = []

        for file_path in files_to_fix:
            full_path = self.project_root / file_path
            if full_path.exists():
                if self._fix_file_paths(full_path):
                    fixed_files.append(str(file_path))

        return fixed_files

    def _fix_file_paths(self, file_path: Path) -> bool:
        """修复单个文件的路径问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 模式1: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))
            pattern1 = r"sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*['\"]\.\.\/\.\.\/\.\./src['\"]\)\)"
            replacement1 = """# 添加项目路径 - 使用pathlib实现跨平台兼容
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))"""

            # 模式2: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
            pattern2 = r"sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*['\"]\.\.\/\.\.\/\.\.\/['\"]\)\)"
            replacement2 = """# 添加项目根目录到路径 - 使用pathlib实现跨平台兼容
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))"""

            content = re.sub(pattern1, replacement1, content)
            content = re.sub(pattern2, replacement2, content)

            # 确保import语句包含pathlib
            if 'from pathlib import Path' in content and 'import sys' in content:
                # 调整import顺序
                content = self._organize_imports(content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

        except Exception as e:
            print(f"❌ 修复文件失败 {file_path}: {e}")

        return False

    def _organize_imports(self, content: str) -> str:
        """整理import语句顺序"""
        lines = content.split('\n')
        organized_lines = []
        imports_section = []
        in_imports = False

        for line in lines:
            stripped = line.strip()

            # 检测import语句
            if (stripped.startswith('import ') or
                stripped.startswith('from ') or
                    (in_imports and stripped == '')):

                if not in_imports:
                    in_imports = True
                imports_section.append(line)

            else:
                if in_imports:
                    # 结束import部分，整理并添加
                    organized_imports = self._sort_imports(imports_section)
                    organized_lines.extend(organized_imports)
                    imports_section = []
                    in_imports = False

                organized_lines.append(line)

        # 处理文件末尾的imports
        if imports_section:
            organized_imports = self._sort_imports(imports_section)
            organized_lines.extend(organized_imports)

        return '\n'.join(organized_lines)

    def _sort_imports(self, imports: List[str]) -> List[str]:
        """排序import语句"""
        # 标准库imports
        stdlib_imports = []
        # 第三方imports
        third_party_imports = []
        # 本地imports
        local_imports = []
        # 空行
        empty_lines = []

        for imp in imports:
            stripped = imp.strip()
            if not stripped:
                empty_lines.append(imp)
                continue

            if stripped.startswith('from pathlib') or stripped.startswith('import sys'):
                stdlib_imports.append(imp)
            elif stripped.startswith('from src.') or stripped.startswith('import src.'):
                local_imports.append(imp)
            else:
                third_party_imports.append(imp)

        # 组合所有imports
        result = []
        if stdlib_imports:
            result.extend(stdlib_imports)
            if third_party_imports or local_imports:
                result.append('')

        if third_party_imports:
            result.extend(third_party_imports)
            if local_imports:
                result.append('')

        if local_imports:
            result.extend(local_imports)

        # 添加尾部空行
        if result and empty_lines:
            result.extend(empty_lines)

        return result

    def fix_platform_specific_code(self) -> List[str]:
        """修复平台特定代码问题"""
        print("🔧 修复平台特定代码问题...")

        # 查找需要修复的文件
        files_to_check = [
            "src/infrastructure/config/security/enhanced_secure_config.py",
            "src/infrastructure/resource/system_monitor.py",
            "src/infrastructure/monitoring/prometheus_compat.py"
        ]

        fixed_files = []

        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                if self._fix_platform_code(full_path):
                    fixed_files.append(str(file_path))

        return fixed_files

    def _fix_platform_code(self, file_path: Path) -> bool:
        """修复单个文件的平台特定代码"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复os.uname()问题
            uname_pattern = r'system_info\s*=\s*str\(os\.uname\(\)\)'
            uname_replacement = '''try:
                system_info = str(os.uname())
            except AttributeError:
                # Windows系统没有uname，使用替代方案
                import platform
                system_info = f"{platform.system()} {platform.release()} {platform.machine()}"'''

            content = re.sub(uname_pattern, uname_replacement, content)

            # 修复平台检测代码
            if 'IS_WINDOWS = platform.system().lower() == "windows"' not in content:
                # 在import section后添加平台检测常量
                import_end = content.find('\n\n')
                if import_end != -1:
                    platform_detection = '''
# 平台兼容性检测
import platform
IS_WINDOWS = platform.system().lower() == "windows"
IS_LINUX = platform.system().lower() == "linux"
IS_MACOS = platform.system().lower() == "darwin"
'''
                    content = content[:import_end] + platform_detection + content[import_end:]

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

        except Exception as e:
            print(f"❌ 修复平台特定代码失败 {file_path}: {e}")

        return False

    def run_fixes(self) -> Dict[str, List[str]]:
        """运行所有修复"""
        print("🚀 开始平台兼容性修复...")

        results = {
            'hardcoded_paths': self.fix_hardcoded_paths(),
            'platform_specific': self.fix_platform_specific_code()
        }

        print(f"✅ 修复完成:")
        print(f"  - 硬编码路径修复: {len(results['hardcoded_paths'])} 个文件")
        print(f"  - 平台特定代码修复: {len(results['platform_specific'])} 个文件")

        return results


if __name__ == "__main__":
    fixer = PlatformCompatibilityFixer()
    results = fixer.run_fixes()

    # 输出详细结果
    if results['hardcoded_paths']:
        print("\n📁 硬编码路径修复文件:")
        for file in results['hardcoded_paths']:
            print(f"  ✓ {file}")

    if results['platform_specific']:
        print("\n🖥️ 平台特定代码修复文件:")
        for file in results['platform_specific']:
            print(f"  ✓ {file}")

    print("\n🎉 平台兼容性修复已完成！")
