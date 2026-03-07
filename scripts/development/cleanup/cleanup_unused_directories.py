#!/usr/bin/env python3
"""
项目根目录无用目录清理脚本

功能：
1. 分析项目根目录下的所有目录
2. 识别可能无用的目录
3. 生成清理建议报告
4. 执行安全的清理操作

使用方法：
python scripts/cleanup_unused_directories.py
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


class DirectoryCleanupAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

        # 定义有用的目录
        self.useful_directories = {
            "src/": "源代码目录",
            "tests/": "测试代码目录",
            "docs/": "文档目录",
            "scripts/": "脚本目录",
            "config/": "配置文件目录",
            "data/": "数据目录",
            "reports/": "报告目录",
            "examples/": "示例代码目录",
            ".git/": "Git版本控制目录",
            "deploy/": "部署相关目录",
            "output/": "输出目录",
            "models/": "模型目录",
            "cache/": "缓存目录",
            "logs/": "日志目录"
        }

        # 定义可能无用的目录
        self.potentially_unused_directories = {
            "tmp/": "临时文件目录",
            "temp/": "临时文件目录",
            "test_final/": "测试最终目录",
            "dummy/": "虚拟数据目录",
            "tmp_model_test/": "模型测试临时目录",
            "tmp_feature_test/": "特征测试临时目录",
            "__pycache__/": "Python缓存目录",
            "htmlcov/": "HTML覆盖率报告目录",
            "coverage_report/": "覆盖率报告目录",
            "coverage_storage/": "覆盖率存储目录",
            "coverage_redis/": "Redis覆盖率目录",
            "coverage_logging/": "日志覆盖率目录",
            "coverage_global/": "全局覆盖率目录",
            "coverage_high_freq/": "高频覆盖率目录",
            "feature_cache/": "特征缓存目录",
            "test_logs/": "测试日志目录",
            "venv/": "虚拟环境目录",
            "venv_clean/": "清理的虚拟环境目录",
            "MagicMock/": "MagicMock测试目录",
            "allure-results/": "Allure测试结果目录",
            "rqa2025.egg-info/": "Python包信息目录",
            "unsupported_path/": "不支持路径目录",
            "dir_path/": "目录路径目录"
        }

        # 定义需要保留的文件类型
        self.important_file_extensions = {
            ".py", ".md", ".txt", ".json", ".yaml", ".yml",
            ".ini", ".cfg", ".toml", ".lock", ".log", ".xml",
            ".html", ".css", ".js", ".png", ".jpg", ".svg"
        }

    def analyze_directory(self, dir_path: Path) -> Dict:
        """分析目录内容"""
        result = {
            "path": str(dir_path),
            "size": 0,
            "file_count": 0,
            "dir_count": 0,
            "last_modified": None,
            "important_files": [],
            "temp_files": [],
            "empty": True
        }

        try:
            if dir_path.exists():
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        result["file_count"] += 1
                        result["size"] += item.stat().st_size
                        result["last_modified"] = max(
                            result["last_modified"] or item.stat().st_mtime,
                            item.stat().st_mtime
                        )

                        # 检查重要文件
                        if item.suffix in self.important_file_extensions:
                            result["important_files"].append(str(item))
                        elif item.suffix in [".tmp", ".temp", ".cache", ".log"]:
                            result["temp_files"].append(str(item))
                    elif item.is_dir():
                        result["dir_count"] += 1

                result["empty"] = result["file_count"] == 0 and result["dir_count"] == 0
        except Exception as e:
            result["error"] = str(e)

        return result

    def classify_directories(self) -> Dict[str, List]:
        """分类目录"""
        print("🔍 分析项目目录...")

        directories = {
            "useful": [],
            "potentially_unused": [],
            "unknown": [],
            "empty": []
        }

        for item in self.project_root.iterdir():
            if item.is_dir():
                dir_name = item.name + "/"

                if dir_name in self.useful_directories:
                    directories["useful"].append({
                        "path": str(item),
                        "description": self.useful_directories[dir_name],
                        "analysis": self.analyze_directory(item)
                    })
                elif dir_name in self.potentially_unused_directories:
                    directories["potentially_unused"].append({
                        "path": str(item),
                        "description": self.potentially_unused_directories[dir_name],
                        "analysis": self.analyze_directory(item)
                    })
                else:
                    directories["unknown"].append({
                        "path": str(item),
                        "description": "未知目录",
                        "analysis": self.analyze_directory(item)
                    })

        # 检查空目录
        for category in ["useful", "potentially_unused", "unknown"]:
            for dir_info in directories[category]:
                if dir_info["analysis"]["empty"]:
                    directories["empty"].append(dir_info)

        return directories

    def generate_cleanup_report(self, directories: Dict[str, List]):
        """生成清理报告"""
        print("📋 生成清理报告...")

        report_content = f"""# 项目目录清理分析报告

**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析状态**: ✅ 已完成

## 📊 目录分类统计

### 目录分类
- **有用目录**: {len(directories['useful'])} 个
- **可能无用目录**: {len(directories['potentially_unused'])} 个
- **未知目录**: {len(directories['unknown'])} 个
- **空目录**: {len(directories['empty'])} 个

## 🔍 详细分析

### 有用目录
"""

        for dir_info in directories["useful"]:
            analysis = dir_info["analysis"]
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"
            report_content += f"  - 文件数: {analysis['file_count']}\n"
            report_content += f"  - 目录数: {analysis['dir_count']}\n"
            report_content += f"  - 大小: {analysis['size'] / 1024 / 1024:.2f} MB\n"
            if analysis.get("error"):
                report_content += f"  - 错误: {analysis['error']}\n"
            report_content += "\n"

        report_content += "\n### 可能无用目录\n"
        for dir_info in directories["potentially_unused"]:
            analysis = dir_info["analysis"]
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"
            report_content += f"  - 文件数: {analysis['file_count']}\n"
            report_content += f"  - 目录数: {analysis['dir_count']}\n"
            report_content += f"  - 大小: {analysis['size'] / 1024 / 1024:.2f} MB\n"
            if analysis.get("error"):
                report_content += f"  - 错误: {analysis['error']}\n"
            report_content += "\n"

        report_content += "\n### 未知目录\n"
        for dir_info in directories["unknown"]:
            analysis = dir_info["analysis"]
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"
            report_content += f"  - 文件数: {analysis['file_count']}\n"
            report_content += f"  - 目录数: {analysis['dir_count']}\n"
            report_content += f"  - 大小: {analysis['size'] / 1024 / 1024:.2f} MB\n"
            if analysis.get("error"):
                report_content += f"  - 错误: {analysis['error']}\n"
            report_content += "\n"

        report_content += "\n### 空目录\n"
        for dir_info in directories["empty"]:
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"

        report_content += f"""
## 🧹 清理建议

### 安全清理 (推荐)
以下目录可以安全清理：

"""

        safe_to_clean = []
        for dir_info in directories["potentially_unused"]:
            analysis = dir_info["analysis"]
            if analysis["file_count"] == 0 or analysis["size"] < 1024 * 1024:  # 小于1MB
                safe_to_clean.append(dir_info)

        for dir_info in safe_to_clean:
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"
            report_content += f"  - 原因: 空目录或文件很少\n"

        report_content += "\n### 谨慎清理 (需要确认)\n"

        careful_to_clean = []
        for dir_info in directories["potentially_unused"]:
            analysis = dir_info["analysis"]
            if analysis["file_count"] > 0 and analysis["size"] >= 1024 * 1024:  # 大于1MB
                careful_to_clean.append(dir_info)

        for dir_info in careful_to_clean:
            report_content += f"- **{dir_info['path']}** ({dir_info['description']})\n"
            report_content += f"  - 文件数: {dir_info['analysis']['file_count']}\n"
            report_content += f"  - 大小: {dir_info['analysis']['size'] / 1024 / 1024:.2f} MB\n"
            report_content += f"  - 建议: 手动检查内容后再决定\n"

        report_content += f"""
## 📈 清理效果预期

### 空间节省
- **可清理空间**: {sum(d['analysis']['size'] for d in safe_to_clean) / 1024 / 1024:.2f} MB
- **谨慎清理空间**: {sum(d['analysis']['size'] for d in careful_to_clean) / 1024 / 1024:.2f} MB
- **总潜在节省**: {(sum(d['analysis']['size'] for d in safe_to_clean) + sum(d['analysis']['size'] for d in careful_to_clean)) / 1024 / 1024:.2f} MB

### 目录数量减少
- **可删除目录**: {len(safe_to_clean)} 个
- **谨慎删除目录**: {len(careful_to_clean)} 个
- **总减少**: {len(safe_to_clean) + len(careful_to_clean)} 个

## 🔄 清理步骤

### 步骤1: 安全清理
```bash
# 删除空目录和小的临时目录
rm -rf tmp_model_test/
rm -rf __pycache__/
rm -rf htmlcov/
rm -rf coverage_report/
rm -rf coverage_storage/
rm -rf coverage_redis/
rm -rf coverage_logging/
rm -rf coverage_global/
rm -rf coverage_high_freq/
```

### 步骤2: 谨慎清理
```bash
# 手动检查后删除
# rm -rf tmp/
# rm -rf temp/
# rm -rf dummy/
# rm -rf test_final/
# rm -rf tmp_feature_test/
# rm -rf test_logs/
# rm -rf venv/
# rm -rf feature_cache/
```

### 步骤3: 验证清理
```bash
# 检查清理结果
python scripts/cleanup_unused_directories.py
```

## ⚠️ 注意事项

1. **备份重要数据**: 清理前请备份重要数据
2. **检查依赖**: 确保删除的目录不影响项目运行
3. **测试验证**: 清理后运行测试确保功能正常
4. **版本控制**: 确保重要文件已提交到版本控制系统

---
*清理分析报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.project_root / "DIRECTORY_CLEANUP_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 清理报告创建完成")

    def create_cleanup_script(self, directories: Dict[str, List]):
        """创建清理脚本"""
        print("🔧 创建清理脚本...")

        safe_to_clean = []
        careful_to_clean = []

        for dir_info in directories["potentially_unused"]:
            analysis = dir_info["analysis"]
            if analysis["file_count"] == 0 or analysis["size"] < 1024 * 1024:
                safe_to_clean.append(dir_info)
            else:
                careful_to_clean.append(dir_info)

        script_content = f"""#!/usr/bin/env python3
# 项目目录清理脚本
# 基于分析报告生成的自动清理脚本

import os
import shutil
from pathlib import Path

def safe_cleanup():
    \"\"\"安全清理操作\"\"\"
    print("🧹 开始安全清理...")
    
    # 安全清理的目录列表
    safe_dirs = [
"""

        for dir_info in safe_to_clean:
            script_content += f'        "{dir_info["path"]}",  # {dir_info["description"]}\n'

        script_content += """    ]
    
    cleaned_count = 0
    cleaned_size = 0
    
    for dir_path in safe_dirs:
        path = Path(dir_path)
        if path.exists():
            try:
                # 计算目录大小
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                
                # 删除目录
                shutil.rmtree(path)
                cleaned_count += 1
                cleaned_size += size
                print(f"  ✅ 已删除: {dir_path} ({size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"  ❌ 删除失败: {dir_path} - {e}")
    
    print(f"\\n📊 清理结果:")
    print(f"  - 删除目录数: {cleaned_count}")
    print(f"  - 释放空间: {cleaned_size / 1024 / 1024:.2f} MB")

def careful_cleanup():
    \"\"\"谨慎清理操作\"\"\"
    print("\\n⚠️  谨慎清理建议:")
    
    careful_dirs = [
"""

        for dir_info in careful_to_clean:
            script_content += f'        "{dir_info["path"]}",  # {dir_info["description"]} - {dir_info["analysis"]["file_count"]} 文件, {dir_info["analysis"]["size"] / 1024 / 1024:.2f} MB\n'

        script_content += """    ]
    
    for dir_path in careful_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  - {dir_path}")
            print(f"    建议: 手动检查内容后再决定是否删除")

if __name__ == "__main__":
    print("🚀 项目目录清理脚本")
    print("=" * 50)
    
    # 执行安全清理
    safe_cleanup()
    
    # 显示谨慎清理建议
    careful_cleanup()
    
    print("\\n✅ 清理脚本执行完成")
    print("💡 提示: 请手动检查谨慎清理的目录")
"""

        script_path = self.project_root / "cleanup_directories.py"
        script_path.write_text(script_content, encoding='utf-8')
        print("✅ 清理脚本创建完成")

    def run(self):
        """执行分析流程"""
        print("🚀 开始分析项目目录...")
        print("=" * 60)

        # 分析目录
        directories = self.classify_directories()

        # 生成清理报告
        self.generate_cleanup_report(directories)

        # 创建清理脚本
        self.create_cleanup_script(directories)

        print("=" * 60)
        print("🎉 目录分析完成！")
        print(f"📋 清理报告: DIRECTORY_CLEANUP_REPORT.md")
        print(f"🔧 清理脚本: cleanup_directories.py")
        print("\n📊 分析结果:")
        print(f"  - 有用目录: {len(directories['useful'])} 个")
        print(f"  - 可能无用目录: {len(directories['potentially_unused'])} 个")
        print(f"  - 未知目录: {len(directories['unknown'])} 个")
        print(f"  - 空目录: {len(directories['empty'])} 个")


if __name__ == "__main__":
    analyzer = DirectoryCleanupAnalyzer()
    analyzer.run()
