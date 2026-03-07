#!/usr/bin/env python3
"""
快速检查8个谨慎清理目录的脚本

功能：
1. 快速检查8个谨慎清理目录的状态
2. 提供详细的目录信息
3. 生成检查建议
4. 记录检查结果

使用方法：
python scripts/check_careful_directories.py
"""

from datetime import datetime
from pathlib import Path
from typing import Dict


class CarefulDirectoryChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

        # 需要检查的目录
        self.careful_directories = {
            "allure-results": {
                "description": "Allure测试结果报告目录",
                "priority": "🔴 高",
                "expected_files": [".html", ".json", ".xml"],
                "check_points": [
                    "是否包含重要的测试失败信息",
                    "是否有需要保留的测试历史记录",
                    "是否包含性能测试结果"
                ]
            },
            "coverage_global": {
                "description": "全局代码覆盖率报告",
                "priority": "🟡 中",
                "expected_files": [".html", ".json", ".xml"],
                "check_points": [
                    "是否包含最新的覆盖率数据",
                    "是否有覆盖率趋势分析",
                    "是否包含重要的覆盖率报告"
                ]
            },
            "coverage_high_freq": {
                "description": "高频测试覆盖率报告",
                "priority": "🟡 中",
                "expected_files": [".html", ".json"],
                "check_points": [
                    "是否包含高频交易模块的覆盖率",
                    "是否有性能相关的覆盖率数据",
                    "是否包含重要的测试结果"
                ]
            },
            "coverage_report": {
                "description": "综合覆盖率报告目录",
                "priority": "🟡 中",
                "expected_files": [".html", ".json", ".xml"],
                "check_points": [
                    "是否包含完整的覆盖率报告",
                    "是否有覆盖率趋势分析",
                    "是否包含重要的覆盖率统计"
                ]
            },
            "htmlcov": {
                "description": "HTML格式的覆盖率报告",
                "priority": "🟡 中",
                "expected_files": [".html", ".css", ".js"],
                "check_points": [
                    "是否包含最新的HTML覆盖率报告",
                    "是否有详细的覆盖率可视化",
                    "是否包含重要的覆盖率分析"
                ]
            },
            "test_logs": {
                "description": "测试日志文件目录",
                "priority": "🟢 低",
                "expected_files": [".log", ".txt"],
                "check_points": [
                    "是否包含重要的测试错误日志",
                    "是否有需要分析的测试问题",
                    "是否包含性能测试日志"
                ]
            },
            "venv": {
                "description": "Python虚拟环境目录",
                "priority": "🔴 高",
                "expected_files": [".py", ".exe", ".dll"],
                "check_points": [
                    "是否为当前使用的虚拟环境",
                    "是否包含项目所需的依赖包",
                    "是否有重要的自定义包"
                ]
            },
            "venv_clean": {
                "description": "清理后的虚拟环境目录",
                "priority": "🟢 低",
                "expected_files": [".py", ".exe"],
                "check_points": [
                    "是否为备用虚拟环境",
                    "是否包含重要的自定义配置",
                    "是否与主环境重复"
                ]
            }
        }

    def analyze_directory(self, dir_path: Path) -> Dict:
        """分析目录内容"""
        result = {
            "exists": False,
            "size": 0,
            "file_count": 0,
            "dir_count": 0,
            "last_modified": None,
            "file_types": {},
            "largest_files": [],
            "recent_files": [],
            "error": None
        }

        try:
            if dir_path.exists():
                result["exists"] = True

                for item in dir_path.rglob("*"):
                    if item.is_file():
                        result["file_count"] += 1
                        result["size"] += item.stat().st_size

                        # 记录文件类型
                        ext = item.suffix.lower()
                        result["file_types"][ext] = result["file_types"].get(ext, 0) + 1

                        # 记录最新修改时间
                        mtime = item.stat().st_mtime
                        if result["last_modified"] is None or mtime > result["last_modified"]:
                            result["last_modified"] = mtime

                        # 记录最大文件
                        result["largest_files"].append((item.stat().st_size, str(item)))

                        # 记录最近文件
                        if mtime > datetime.now().timestamp() - 7 * 24 * 3600:  # 7天内
                            result["recent_files"].append((mtime, str(item)))

                    elif item.is_dir():
                        result["dir_count"] += 1

                # 排序最大文件和最近文件
                result["largest_files"].sort(reverse=True)
                result["recent_files"].sort(reverse=True)
                result["largest_files"] = result["largest_files"][:5]
                result["recent_files"] = result["recent_files"][:5]

        except Exception as e:
            result["error"] = str(e)

        return result

    def check_virtual_environment(self, dir_path: Path) -> Dict:
        """检查虚拟环境"""
        result = {
            "is_current": False,
            "python_version": None,
            "pip_packages": [],
            "size": 0
        }

        try:
            if dir_path.exists():
                # 检查Python可执行文件
                python_exe = dir_path / "Scripts" / "python.exe"  # Windows
                if not python_exe.exists():
                    python_exe = dir_path / "bin" / "python"  # Unix

                if python_exe.exists():
                    # 检查是否为当前环境
                    import sys
                    current_python = sys.executable
                    result["is_current"] = str(python_exe) in current_python

                    # 获取Python版本
                    try:
                        import subprocess
                        version_output = subprocess.check_output([str(python_exe), "--version"],
                                                                 capture_output=True, text=True)
                        result["python_version"] = version_output.strip()
                    except:
                        result["python_version"] = "无法获取版本"

                    # 获取包列表
                    try:
                        pip_exe = dir_path / "Scripts" / "pip.exe"
                        if not pip_exe.exists():
                            pip_exe = dir_path / "bin" / "pip"

                        if pip_exe.exists():
                            packages_output = subprocess.check_output([str(pip_exe), "list"],
                                                                      capture_output=True, text=True)
                            result["pip_packages"] = packages_output.strip().split('\n')[:10]
                    except:
                        result["pip_packages"] = ["无法获取包列表"]

                # 计算大小
                result["size"] = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

        except Exception as e:
            result["error"] = str(e)

        return result

    def generate_check_report(self, analysis_results: Dict):
        """生成检查报告"""
        print("📋 生成检查报告...")

        report_content = f"""# 8个谨慎清理目录检查报告

**检查时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**检查状态**: ✅ 已完成

## 📊 检查概览

### 目录检查统计
- **总目录数**: 8个
- **存在目录**: {sum(1 for result in analysis_results.values() if result['exists'])}个
- **不存在目录**: {sum(1 for result in analysis_results.values() if not result['exists'])}个

## 🔍 详细检查结果

"""

        for dir_name, dir_info in self.careful_directories.items():
            analysis = analysis_results[dir_name]
            report_content += f"### {dir_name}/ ({dir_info['description']})\n"
            report_content += f"**优先级**: {dir_info['priority']}\n\n"

            if analysis['exists']:
                size_mb = analysis['size'] / 1024 / 1024
                report_content += f"- **状态**: ✅ 存在\n"
                report_content += f"- **大小**: {size_mb:.2f} MB\n"
                report_content += f"- **文件数**: {analysis['file_count']}\n"
                report_content += f"- **目录数**: {analysis['dir_count']}\n"

                if analysis['last_modified']:
                    last_modified = datetime.fromtimestamp(analysis['last_modified'])
                    report_content += f"- **最后修改**: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n"

                if analysis['file_types']:
                    report_content += f"- **主要文件类型**: {', '.join(list(analysis['file_types'].keys())[:5])}\n"

                if analysis['largest_files']:
                    report_content += f"- **最大文件**: {analysis['largest_files'][0][1] if analysis['largest_files'] else '无'}\n"

                if analysis['recent_files']:
                    report_content += f"- **最近文件**: {analysis['recent_files'][0][1] if analysis['recent_files'] else '无'}\n"

                # 特殊检查
                if dir_name in ['venv', 'venv_clean']:
                    venv_info = self.check_virtual_environment(self.project_root / dir_name)
                    if venv_info['is_current']:
                        report_content += f"- **虚拟环境**: 🔴 当前使用的环境\n"
                    else:
                        report_content += f"- **虚拟环境**: ⚪ 非当前环境\n"

                    if venv_info['python_version']:
                        report_content += f"- **Python版本**: {venv_info['python_version']}\n"

                report_content += f"\n**检查要点**:\n"
                for point in dir_info['check_points']:
                    report_content += f"- {point}\n"

            else:
                report_content += f"- **状态**: ❌ 不存在\n"

            report_content += "\n---\n\n"

        report_content += f"""
## 💡 检查建议

### 高优先级目录 (🔴)
1. **allure-results/**: 检查是否包含重要的测试失败分析
2. **venv/**: 确认是否为当前使用的虚拟环境

### 中优先级目录 (🟡)
3. **coverage_global/**: 检查覆盖率报告的重要性
4. **coverage_high_freq/**: 检查高频测试覆盖率
5. **coverage_report/**: 检查综合覆盖率报告
6. **htmlcov/**: 检查HTML覆盖率报告

### 低优先级目录 (🟢)
7. **test_logs/**: 检查测试日志的重要性
8. **venv_clean/**: 检查备用环境状态

## 🔧 操作建议

### 可以安全删除的情况
- 目录不存在或为空
- 只包含临时文件且可以重新生成
- 历史数据且不再需要
- 重复的虚拟环境

### 需要保留的情况
- 包含重要的测试失败信息
- 包含覆盖率趋势分析
- 当前使用的虚拟环境
- 包含重要的调试信息

### 建议操作步骤
1. **备份重要数据**: 删除前备份重要文件
2. **分批删除**: 一次删除1-2个目录
3. **验证功能**: 删除后运行测试
4. **记录操作**: 记录删除的目录和原因

## 📝 检查清单

"""

        for dir_name in self.careful_directories.keys():
            analysis = analysis_results[dir_name]
            status = "✅ 存在" if analysis['exists'] else "❌ 不存在"
            report_content += f"- [ ] {dir_name}/ - {status}\n"

        report_content += f"""
---
*检查报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.project_root / "CAREFUL_DIRECTORIES_CHECK_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 检查报告创建完成")

    def run(self):
        """执行检查流程"""
        print("🔍 开始检查8个谨慎清理目录...")
        print("=" * 60)

        # 分析所有目录
        analysis_results = {}
        for dir_name in self.careful_directories.keys():
            dir_path = self.project_root / dir_name
            print(f"  📁 检查 {dir_name}/...")
            analysis_results[dir_name] = self.analyze_directory(dir_path)

        # 生成检查报告
        self.generate_check_report(analysis_results)

        print("=" * 60)
        print("🎉 目录检查完成！")
        print(f"📋 检查报告: CAREFUL_DIRECTORIES_CHECK_REPORT.md")

        # 显示检查结果摘要
        print("\n📊 检查结果摘要:")
        for dir_name, analysis in analysis_results.items():
            if analysis['exists']:
                size_mb = analysis['size'] / 1024 / 1024
                print(f"  - {dir_name}/: {size_mb:.2f} MB, {analysis['file_count']} 文件")
            else:
                print(f"  - {dir_name}/: 目录不存在")


if __name__ == "__main__":
    checker = CarefulDirectoryChecker()
    checker.run()
