#!/usr/bin/env python3
"""
项目根目录文件清理脚本

功能：
1. 分析项目根目录下的所有文件
2. 识别可能无用的文件
3. 生成清理建议报告
4. 执行安全的清理操作

使用方法：
python scripts/cleanup_root_files.py
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


class RootFileCleanupAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

        # 定义有用的文件
        self.useful_files = {
            "README.md": "项目说明文档",
            "requirements.txt": "Python依赖文件",
            "requirements.lock": "Python依赖锁定文件",
            "pyproject.toml": "Python项目配置文件",
            "pytest.ini": "Pytest配置文件",
            ".coveragerc": "覆盖率配置文件",
            ".pre-commit-config.yaml": "Pre-commit配置文件",
            "pydoc-markdown.yml": "文档生成配置",
            "conda_env_init.ps1": "Conda环境初始化脚本",
            "run-pre-commit.ps1": "Pre-commit运行脚本",
            "TECHNICAL_DEBT.md": "技术债务文档",
            "auto_tech_debt.py": "技术债务自动化脚本",
            "check_coverage.py": "覆盖率检查脚本",
            "check_env.py": "环境检查脚本",
            "check_powershell_env.py": "PowerShell环境检查脚本",
            "run_tests.py": "测试运行脚本",
            "run_tests_powershell.py": "PowerShell测试运行脚本",
            "run_tests_powershell.ps1": "PowerShell测试脚本",
            "run_tests_cmd.bat": "CMD测试脚本",
            "run_optimized_tests.py": "优化测试运行脚本",
            "run_fast_tests.py": "快速测试运行脚本",
            "run_test.py": "单次测试运行脚本",
            "fix_technical_tests.py": "技术测试修复脚本",
            "fix_technical_tests_v2.py": "技术测试修复脚本v2",
            "fix_technical_tests_v3.py": "技术测试修复脚本v3",
            "fix_deadlock_tests.py": "死锁测试修复脚本",
            "direct_test.py": "直接测试脚本",
            "create_test_dirs.py": "测试目录创建脚本",
            "conftest.py": "Pytest配置文件",
            "test_miniqmt_enhancement.py": "MiniQMT增强测试",
            "test_acceleration_structure.py": "加速结构测试",
            "test_directory_structure.py": "目录结构测试",
            "test_optimization_complete.py": "优化完成测试",
            "test_module_reorganization.py": "模块重组测试",
            "test_import_config.py": "导入配置测试",
            "test_import.py": "导入测试",
            "test_error_handler_comprehensive.py": "错误处理综合测试",
            "test_env.py": "环境测试",
            "temp_log_test.py": "临时日志测试",
            "transformers_test.py": "Transformers测试",
            "src_directory_structure_analysis.md": "源代码目录结构分析",
            "batch_automation_state.json": "批处理自动化状态",
            "model_landing.log": "模型落地日志",
            "model_landing_advanced.log": "高级模型落地日志",
            "focused_test_advancement.log": "重点测试进展日志",
            "batch_model_landing.log": "批处理模型落地日志",
            "test_app.json.log": "测试应用JSON日志",
            "coverage.json": "覆盖率JSON数据",
            "rqa2025_temp.db": "临时数据库文件",
            "dummy.pkl": "虚拟数据文件"
        }

        # 定义可能无用的文件
        self.potentially_unused_files = {
            # 覆盖率相关文件
            ".coverage": "覆盖率数据文件",
            "coverage.xml": "覆盖率XML报告",
            "coverage_step1.txt": "覆盖率步骤1文件",
            "portfolio_cov.txt": "投资组合覆盖率文件",
            "data_cov.txt": "数据覆盖率文件",
            "features_cov.txt": "特征覆盖率文件",
            "fpga_cov.txt": "FPGA覆盖率文件",
            "ensemble_cov.txt": "集成覆盖率文件",
            "ensemble_coverage.txt": "集成覆盖率文件",
            "coverage_final.txt": "最终覆盖率文件",
            "coverage_final_2.txt": "最终覆盖率文件2",
            "coverage_updated.txt": "更新的覆盖率文件",
            "coverage_ensemble_fix.txt": "集成覆盖率修复文件",
            "coverage_report.txt": "覆盖率报告文件",
            "lowcov.txt": "低覆盖率文件",

            # 测试结果文件
            "detailed_test_results.xml": "详细测试结果XML",
            "ensemble_test_results.xml": "集成测试结果XML",
            "test-results.xml": "测试结果XML",

            # 测试日志文件
            "pytest_ensemble_coverage.log": "Pytest集成覆盖率日志",
            "pytest_model_ensemble_coverage.log": "Pytest模型集成覆盖率日志",
            "pytest_parameter_optimizer.log": "Pytest参数优化器日志",
            "pytest_features_coverage.log": "Pytest特征覆盖率日志",
            "pytest_coverage.log": "Pytest覆盖率日志",
            "pytest_fpga_features.log": "Pytest FPGA特征日志",
            "pytest_complete_results.log": "Pytest完整结果日志",
            "pytest_post_fix_results.log": "Pytest修复后结果日志",
            "pytest_collection_errors.log": "Pytest收集错误日志",
            "pytest_collection_detailed.log": "Pytest详细收集日志",
            "pytest_collection_errors.log": "Pytest收集错误日志",
            "pytest_debug.log": "Pytest调试日志",
            "pytest_detailed.log": "Pytest详细日志",

            # 临时文件
            "terminal.integrated.profiles.windows": "终端集成配置文件",

            # 清理报告文件
            "DIRECTORY_CLEANUP_REPORT.md": "目录清理报告",
            "DIRECTORY_CLEANUP_COMPLETION_REPORT.md": "目录清理完成报告",
            "CAREFUL_DIRECTORIES_CHECK_REPORT.md": "谨慎目录检查报告",
            "MANUAL_DIRECTORY_CHECK_GUIDE.md": "手工目录检查指南",
            "cleanup_directories.py": "目录清理脚本"
        }

        # 定义需要保留的文件类型
        self.important_file_extensions = {
            ".py", ".md", ".txt", ".json", ".yaml", ".yml",
            ".ini", ".cfg", ".toml", ".lock", ".log", ".xml",
            ".html", ".css", ".js", ".png", ".jpg", ".svg",
            ".ps1", ".bat", ".sh", ".yml", ".yaml"
        }

    def analyze_file(self, file_path: Path) -> Dict:
        """分析文件内容"""
        result = {
            "path": str(file_path),
            "size": 0,
            "last_modified": None,
            "extension": file_path.suffix.lower(),
            "is_binary": False,
            "line_count": 0,
            "error": None
        }

        try:
            if file_path.exists():
                stat = file_path.stat()
                result["size"] = stat.st_size
                result["last_modified"] = stat.st_mtime

                # 检查是否为二进制文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1024)  # 只读取前1024字节
                        result["line_count"] = content.count('\n')
                        result["is_binary"] = '\x00' in content
                except UnicodeDecodeError:
                    result["is_binary"] = True
                except Exception as e:
                    result["error"] = str(e)

        except Exception as e:
            result["error"] = str(e)

        return result

    def classify_files(self) -> Dict[str, List]:
        """分类文件"""
        print("🔍 分析项目根目录文件...")

        files = {
            "useful": [],
            "potentially_unused": [],
            "unknown": [],
            "large_files": []
        }

        for item in self.project_root.iterdir():
            if item.is_file():
                file_name = item.name
                analysis = self.analyze_file(item)

                if file_name in self.useful_files:
                    files["useful"].append({
                        "name": file_name,
                        "description": self.useful_files[file_name],
                        "analysis": analysis
                    })
                elif file_name in self.potentially_unused_files:
                    files["potentially_unused"].append({
                        "name": file_name,
                        "description": self.potentially_unused_files[file_name],
                        "analysis": analysis
                    })
                else:
                    files["unknown"].append({
                        "name": file_name,
                        "description": "未知文件",
                        "analysis": analysis
                    })

                # 检查大文件
                if analysis["size"] > 1024 * 1024:  # 大于1MB
                    files["large_files"].append({
                        "name": file_name,
                        "description": "大文件",
                        "analysis": analysis
                    })

        return files

    def generate_cleanup_report(self, files: Dict[str, List]):
        """生成清理报告"""
        print("📋 生成清理报告...")

        report_content = f"""# 项目根目录文件清理分析报告

**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析状态**: ✅ 已完成

## 📊 文件分类统计

### 文件分类
- **有用文件**: {len(files['useful'])} 个
- **可能无用文件**: {len(files['potentially_unused'])} 个
- **未知文件**: {len(files['unknown'])} 个
- **大文件**: {len(files['large_files'])} 个

## 🔍 详细分析

### 有用文件
"""

        for file_info in files["useful"]:
            analysis = file_info["analysis"]
            size_mb = analysis["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']})\n"
            report_content += f"  - 大小: {size_mb:.2f} MB\n"
            if analysis["last_modified"]:
                last_modified = datetime.fromtimestamp(analysis["last_modified"])
                report_content += f"  - 最后修改: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += "\n"

        report_content += "\n### 可能无用文件\n"
        for file_info in files["potentially_unused"]:
            analysis = file_info["analysis"]
            size_mb = analysis["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']})\n"
            report_content += f"  - 大小: {size_mb:.2f} MB\n"
            if analysis["last_modified"]:
                last_modified = datetime.fromtimestamp(analysis["last_modified"])
                report_content += f"  - 最后修改: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += "\n"

        report_content += "\n### 未知文件\n"
        for file_info in files["unknown"]:
            analysis = file_info["analysis"]
            size_mb = analysis["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']})\n"
            report_content += f"  - 大小: {size_mb:.2f} MB\n"
            if analysis["last_modified"]:
                last_modified = datetime.fromtimestamp(analysis["last_modified"])
                report_content += f"  - 最后修改: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += "\n"

        report_content += "\n### 大文件 (>1MB)\n"
        for file_info in files["large_files"]:
            analysis = file_info["analysis"]
            size_mb = analysis["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']})\n"
            report_content += f"  - 大小: {size_mb:.2f} MB\n"
            if analysis["last_modified"]:
                last_modified = datetime.fromtimestamp(analysis["last_modified"])
                report_content += f"  - 最后修改: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += "\n"

        report_content += f"""
## 🧹 清理建议

### 安全清理 (推荐)
以下文件可以安全清理：

"""

        safe_to_clean = []
        for file_info in files["potentially_unused"]:
            analysis = file_info["analysis"]
            if analysis["size"] < 1024 * 1024 or analysis["extension"] in [".log", ".txt", ".xml"]:
                safe_to_clean.append(file_info)

        for file_info in safe_to_clean:
            size_mb = file_info["analysis"]["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']}) - {size_mb:.2f} MB\n"

        report_content += "\n### 谨慎清理 (需要确认)\n"

        careful_to_clean = []
        for file_info in files["potentially_unused"]:
            analysis = file_info["analysis"]
            if analysis["size"] >= 1024 * 1024 and analysis["extension"] not in [".log", ".txt", ".xml"]:
                careful_to_clean.append(file_info)

        for file_info in careful_to_clean:
            size_mb = file_info["analysis"]["size"] / 1024 / 1024
            report_content += f"- **{file_info['name']}** ({file_info['description']}) - {size_mb:.2f} MB\n"
            report_content += f"  - 建议: 手动检查内容后再决定\n"

        report_content += f"""
## 📈 清理效果预期

### 空间节省
- **可清理空间**: {sum(f['analysis']['size'] for f in safe_to_clean) / 1024 / 1024:.2f} MB
- **谨慎清理空间**: {sum(f['analysis']['size'] for f in careful_to_clean) / 1024 / 1024:.2f} MB
- **总潜在节省**: {(sum(f['analysis']['size'] for f in safe_to_clean) + sum(f['analysis']['size'] for f in careful_to_clean)) / 1024 / 1024:.2f} MB

### 文件数量减少
- **可删除文件**: {len(safe_to_clean)} 个
- **谨慎删除文件**: {len(careful_to_clean)} 个
- **总减少**: {len(safe_to_clean) + len(careful_to_clean)} 个

## 🔄 清理步骤

### 步骤1: 安全清理
```bash
# 删除小的临时文件和日志文件
rm .coverage
rm coverage.xml
rm coverage_step1.txt
rm portfolio_cov.txt
rm data_cov.txt
rm features_cov.txt
rm fpga_cov.txt
rm ensemble_cov.txt
rm ensemble_coverage.txt
rm coverage_final.txt
rm coverage_final_2.txt
rm coverage_updated.txt
rm coverage_ensemble_fix.txt
rm coverage_report.txt
rm lowcov.txt
rm detailed_test_results.xml
rm ensemble_test_results.xml
rm test-results.xml
rm pytest_*.log
rm terminal.integrated.profiles.windows
rm DIRECTORY_CLEANUP_REPORT.md
rm DIRECTORY_CLEANUP_COMPLETION_REPORT.md
rm CAREFUL_DIRECTORIES_CHECK_REPORT.md
rm MANUAL_DIRECTORY_CHECK_GUIDE.md
rm cleanup_directories.py
```

### 步骤2: 谨慎清理
```bash
# 手动检查后删除
# rm coverage_step1.txt  # 25MB
# rm portfolio_cov.txt   # 31MB
# rm data_cov.txt        # 20MB
# rm features_cov.txt    # 20MB
# rm fpga_cov.txt        # 38MB
# rm ensemble_cov.txt    # 31MB
# rm ensemble_coverage.txt # 35MB
# rm pytest_fpga_features.log # 357KB
# rm detailed_test_results.xml # 2.6MB
# rm test_collection_detailed.log # 816KB
# rm test_collection_errors.log # 816KB
# rm pytest_complete_results.log # 1.4MB
# rm pytest_post_fix_results.log # 1.4MB
# rm pytest_collection_errors.log # 6.6MB
# rm pytest_debug.log # 82KB
# rm pytest_detailed.log # 82KB
```

### 步骤3: 验证清理
```bash
# 检查清理结果
python scripts/cleanup_root_files.py
```

## ⚠️ 注意事项

1. **备份重要数据**: 清理前请备份重要文件
2. **检查依赖**: 确保删除的文件不影响项目运行
3. **测试验证**: 清理后运行测试确保功能正常
4. **版本控制**: 确保重要文件已提交到版本控制系统

---
*清理分析报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.project_root / "ROOT_FILES_CLEANUP_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 清理报告创建完成")

    def create_cleanup_script(self, files: Dict[str, List]):
        """创建清理脚本"""
        print("🔧 创建清理脚本...")

        safe_to_clean = []
        careful_to_clean = []

        for file_info in files["potentially_unused"]:
            analysis = file_info["analysis"]
            if analysis["size"] < 1024 * 1024 or analysis["extension"] in [".log", ".txt", ".xml"]:
                safe_to_clean.append(file_info)
            else:
                careful_to_clean.append(file_info)

        script_content = f"""#!/usr/bin/env python3
# 项目根目录文件清理脚本
# 基于分析报告生成的自动清理脚本

import os
import shutil
from pathlib import Path

def safe_cleanup():
    \"\"\"安全清理操作\"\"\"
    print("🧹 开始安全清理...")
    
    # 安全清理的文件列表
    safe_files = [
"""

        for file_info in safe_to_clean:
            size_mb = file_info["analysis"]["size"] / 1024 / 1024
            script_content += f'        "{file_info["name"]}",  # {file_info["description"]} - {size_mb:.2f} MB\n'

        script_content += """    ]
    
    cleaned_count = 0
    cleaned_size = 0
    
    for file_name in safe_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                # 计算文件大小
                size = file_path.stat().st_size
                
                # 删除文件
                file_path.unlink()
                cleaned_count += 1
                cleaned_size += size
                print(f"  ✅ 已删除: {file_name} ({size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"  ❌ 删除失败: {file_name} - {e}")
    
    print(f"\\n📊 清理结果:")
    print(f"  - 删除文件数: {cleaned_count}")
    print(f"  - 释放空间: {cleaned_size / 1024 / 1024:.2f} MB")

def careful_cleanup():
    \"\"\"谨慎清理操作\"\"\"
    print("\\n⚠️  谨慎清理建议:")
    
    careful_files = [
"""

        for file_info in careful_to_clean:
            size_mb = file_info["analysis"]["size"] / 1024 / 1024
            script_content += f'        "{file_info["name"]}",  # {file_info["description"]} - {size_mb:.2f} MB\n'

        script_content += """    ]
    
    for file_name in careful_files:
        file_path = Path(file_name)
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  - {file_name} ({size_mb:.2f} MB)")
            print(f"    建议: 手动检查内容后再决定是否删除")

if __name__ == "__main__":
    print("🚀 项目根目录文件清理脚本")
    print("=" * 50)
    
    # 执行安全清理
    safe_cleanup()
    
    # 显示谨慎清理建议
    careful_cleanup()
    
    print("\\n✅ 清理脚本执行完成")
    print("💡 提示: 请手动检查谨慎清理的文件")
"""

        script_path = self.project_root / "cleanup_root_files.py"
        script_path.write_text(script_content, encoding='utf-8')
        print("✅ 清理脚本创建完成")

    def run(self):
        """执行分析流程"""
        print("🚀 开始分析项目根目录文件...")
        print("=" * 60)

        # 分析文件
        files = self.classify_files()

        # 生成清理报告
        self.generate_cleanup_report(files)

        # 创建清理脚本
        self.create_cleanup_script(files)

        print("=" * 60)
        print("🎉 文件分析完成！")
        print(f"📋 清理报告: ROOT_FILES_CLEANUP_REPORT.md")
        print(f"🔧 清理脚本: cleanup_root_files.py")
        print("\n📊 分析结果:")
        print(f"  - 有用文件: {len(files['useful'])} 个")
        print(f"  - 可能无用文件: {len(files['potentially_unused'])} 个")
        print(f"  - 未知文件: {len(files['unknown'])} 个")
        print(f"  - 大文件: {len(files['large_files'])} 个")


if __name__ == "__main__":
    analyzer = RootFileCleanupAnalyzer()
    analyzer.run()
