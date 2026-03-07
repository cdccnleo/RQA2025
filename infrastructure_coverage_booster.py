#!/usr/bin/env python3
"""
基础设施层覆盖率提升执行器

按阶段系统性地提升基础设施层的测试覆盖率至80%
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import time

class InfrastructureCoverageBooster:
    """基础设施层覆盖率提升器"""

    def __init__(self, output_dir="test_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.modules = {
            'p0': ['config', 'cache', 'health', 'resource', 'security', 'monitoring', 'core', 'interfaces'],
            'p1': ['logging', 'utils', 'api', 'error'],
            'p2': ['versioning', 'distributed', 'optimization', 'constants', 'ops']
        }

    def run_phase1_core_modules(self):
        """Phase 1: 核心模块修复与测试"""
        print("🚀 Phase 1: 核心模块修复与测试")
        print("=" * 50)

        # 1. 修复测试文件语法错误
        print("1. 修复测试文件语法错误...")
        self.fix_test_syntax_errors()

        # 2. 测试核心模块
        print("2. 测试核心模块...")
        results = {}
        for module in self.modules['p0'][:3]:  # 先测试前3个核心模块
            print(f"   测试模块: {module}")
            result = self.test_module_coverage(module)
            results[module] = result
            time.sleep(1)  # 避免并发问题

        # 3. 生成阶段报告
        self.generate_phase_report("phase1", results)

        return results

    def fix_test_syntax_errors(self):
        """修复测试文件的语法错误"""
        infrastructure_dir = Path('tests/unit/infrastructure')

        fixed_count = 0
        total_count = 0

        for py_file in infrastructure_dir.rglob('*.py'):
            if py_file.name.startswith('test_'):
                total_count += 1
                if self.fix_file_syntax(py_file):
                    fixed_count += 1

        print(f"   修复完成: {fixed_count}/{total_count} 个文件")

    def fix_file_syntax(self, file_path):
        """修复单个文件的语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否需要修复
            if '# 确保Python路径正确配置' in content and 'project_root = Path(__file__).resolve()' in content:
                # 检查语法
                try:
                    compile(content, str(file_path), 'exec')
                    return False  # 语法正确，无需修复
                except SyntaxError:
                    pass  # 需要修复

            # 重新生成文件头部
            header = '''"""
基础设施层测试文件

自动修复导入问题
"""

import pytest
import sys
import importlib
from pathlib import Path

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

'''

            # 清理旧内容
            lines = content.split('\n')
            # 移除旧的导入和头部
            cleaned_lines = []
            skip_mode = False
            for line in lines:
                # 跳过旧的头部内容
                if line.strip().startswith('"""') and '基础设施层测试文件' in content:
                    skip_mode = True
                    continue
                if skip_mode and line.strip().startswith('"""'):
                    skip_mode = False
                    continue
                if skip_mode:
                    continue

                # 跳过重复的导入
                if any(phrase in line for phrase in [
                    'import pytest', 'import sys', 'import importlib', 'from pathlib import Path',
                    '# 确保Python路径正确配置', 'project_root = Path', 'src_path_str =',
                    'sys.path.insert', '# 动态导入模块'
                ]):
                    continue

                # 跳过损坏的try块
                if line.strip() == 'try:' and not cleaned_lines:
                    continue

                cleaned_lines.append(line)

            # 重组内容
            new_content = header + '\n'.join(cleaned_lines)

            # 再次检查语法
            try:
                compile(new_content, str(file_path), 'exec')
            except SyntaxError as e:
                print(f"   语法错误仍存在: {file_path} - {e}")
                return False

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return True

        except Exception as e:
            print(f"   修复失败: {file_path} - {e}")
            return False

    def test_module_coverage(self, module_name):
        """测试单个模块的覆盖率"""
        try:
            # 构建测试路径
            test_path = f"tests/unit/infrastructure/{module_name}"

            if not os.path.exists(test_path):
                return {"status": "not_found", "message": f"测试路径不存在: {test_path}"}

            # 运行覆盖率测试
            cmd = [
                sys.executable, "-m", "pytest",
                test_path,
                "--cov", f"src.infrastructure.{module_name}",
                "--cov-report", "term-missing",
                "--cov-report", f"json:{self.output_dir}/coverage_{module_name}.json",
                "-x", "--tb=short", "-q", "--disable-warnings"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            test_result = {
                "return_code": result.returncode,
                "coverage": self.parse_coverage_output(result.stdout),
                "stdout": result.stdout[-500:],  # 只保留最后500字符
                "stderr": result.stderr[-500:] if result.stderr else ""
            }

            if result.returncode == 0:
                test_result["status"] = "success"
            elif "SKIPPED" in result.stdout:
                test_result["status"] = "skipped"
            else:
                test_result["status"] = "failed"

            return test_result

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": "测试超时"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def parse_coverage_output(self, output):
        """解析覆盖率输出"""
        coverage_info = {}

        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        coverage_info = {
                            "statements": int(parts[1]),
                            "missing": int(parts[2]),
                            "coverage_percent": float(parts[3].strip('%'))
                        }
                    except (ValueError, IndexError):
                        pass
                break

        return coverage_info

    def generate_phase_report(self, phase_name, results):
        """生成阶段报告"""
        report_file = self.output_dir / f"infrastructure_{phase_name}_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 基础设施层覆盖率 - {phase_name.upper()} 阶段报告\n\n")
            f.write(f"**生成时间**: {datetime.now().isoformat()}\n\n")

            f.write("## 📊 测试结果\n\n")

            total_modules = len(results)
            successful_modules = sum(1 for r in results.values() if r.get('status') == 'success')
            total_coverage = 0
            coverage_count = 0

            for module, result in results.items():
                f.write(f"### {module}\n\n")
                f.write(f"- **状态**: {result.get('status', 'unknown')}\n")

                coverage = result.get('coverage', {})
                if coverage:
                    pct = coverage.get('coverage_percent', 0)
                    f.write(f"- **覆盖率**: {pct}%\n")
                    f.write(f"- **语句数**: {coverage.get('statements', 0)}\n")
                    f.write(f"- **未覆盖**: {coverage.get('missing', 0)}\n")

                    total_coverage += pct
                    coverage_count += 1

                if result.get('status') not in ['success', 'skipped']:
                    stderr = result.get('stderr', '')
                    if stderr:
                        f.write(f"- **错误**: {stderr[:100]}...\n")

                f.write("\n")

            # 统计信息
            f.write("## 📈 统计信息\n\n")
            if total_modules > 0:
                success_rate = (successful_modules / total_modules) * 100
                f.write(".1f")
            if coverage_count > 0:
                avg_coverage = total_coverage / coverage_count
                f.write(".1f")

            # 改进建议
            f.write("\n## 🎯 改进建议\n\n")
            if successful_modules < total_modules:
                f.write("### 需要修复的模块\n\n")
                for module, result in results.items():
                    if result.get('status') != 'success':
                        f.write(f"- **{module}**: {result.get('status', 'unknown')}\n")
                        if result.get('message'):
                            f.write(f"  - {result.get('message')}\n")

            f.write("\n### 下一阶段计划\n\n")
            if phase_name == "phase1":
                f.write("1. 继续修复剩余P0模块的测试问题\n")
                f.write("2. 开始P1模块的覆盖率提升\n")
                f.write("3. 建立每日覆盖率监控机制\n")

        print(f"📄 阶段报告已生成: {report_file}")

        # 保存JSON结果
        json_file = self.output_dir / f"infrastructure_{phase_name}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "phase": phase_name,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    booster = InfrastructureCoverageBooster()

    print("🏗️ 基础设施层覆盖率提升启动")
    print("=" * 60)

    # 执行Phase 1
    results = booster.run_phase1_core_modules()

    # 输出总结
    print("\n" + "=" * 60)
    print("📋 Phase 1 执行总结")

    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    total = len(results)

    print(f"✅ 成功模块: {successful}/{total}")
    print(".1f")

    if successful < total:
        print("⚠️  仍需修复的模块:")
        for module, result in results.items():
            if result.get('status') != 'success':
                print(f"   - {module}: {result.get('status')}")

    print("\n🚀 下一阶段: 继续修复剩余模块并提升覆盖率")
    print("💡 建议: 重点关注模块导入问题和测试语法错误")

if __name__ == "__main__":
    main()
