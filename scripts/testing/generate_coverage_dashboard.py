#!/usr/bin/env python3
"""
RQA2025 覆盖率仪表板生成器
生成可视化的覆盖率报告和趋势分析
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class CoverageDashboardGenerator:
    """覆盖率仪表板生成器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports" / "testing"
        self.dashboard_dir = self.reports_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

    def load_coverage_history(self) -> List[Dict]:
        """加载覆盖率历史数据"""
        history = []

        if not self.reports_dir.exists():
            return history

        json_files = list(self.reports_dir.glob("coverage_results_*.json"))
        json_files.sort()

        for json_file in json_files[-30:]:  # 最近30次
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取时间戳
                timestamp_str = json_file.stem.split('_', 2)[2]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # 计算平均覆盖率
                total_coverage = 0
                valid_modules = 0

                for module, result in data.items():
                    if result.get("success", False):
                        coverage = self._extract_coverage_from_stdout(result.get("stdout", ""))
                        if coverage > 0:
                            total_coverage += coverage
                            valid_modules += 1

                avg_coverage = total_coverage / valid_modules if valid_modules > 0 else 0

                history.append({
                    "timestamp": timestamp,
                    "average_coverage": avg_coverage,
                    "data": data
                })

            except Exception as e:
                print(f"警告: 无法加载 {json_file}: {e}")

        return history

    def _extract_coverage_from_stdout(self, stdout: str) -> float:
        """从测试输出中提取覆盖率"""
        try:
            lines = stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[3].replace('%', '')
                        return float(coverage_str)
        except:
            pass
        return 0.0

    def generate_html_dashboard(self, history: List[Dict]):
        """生成HTML仪表板"""
        if not history:
            print("⚠️  没有历史数据，跳过HTML仪表板生成")
            return

        latest = history[-1]
        latest_data = latest["data"]

        # 计算各模块覆盖率
        module_coverage = {}
        for module, result in latest_data.items():
            if result.get("success", False):
                coverage = self._extract_coverage_from_stdout(result.get("stdout", ""))
                module_coverage[module] = coverage
            else:
                module_coverage[module] = 0

        # 生成HTML内容
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 测试覆盖率仪表板</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .module-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .module-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
        }}
        .coverage-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 RQA2025 测试覆盖率仪表板</h1>
            <p>最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{latest['average_coverage']:.1f}%</div>
                    <div class="stat-label">平均覆盖率</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(module_coverage)}</div>
                    <div class="stat-label">测试模块数</div>
                </div>
            </div>
            
            <h3>📈 各模块覆盖率详情</h3>
            <div class="module-grid">
"""

        for module, coverage in module_coverage.items():
            html_content += f"""
                <div class="module-card">
                    <div class="module-name">{module.title()}</div>
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {coverage}%"></div>
                    </div>
                    <div class="coverage-text">{coverage:.1f}%</div>
                </div>
"""

        html_content += """
            </div>
        </div>
    </div>
</body>
</html>"""

        # 保存HTML文件
        with open(self.dashboard_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("✅ HTML仪表板已生成")

    def generate_dashboard(self):
        """生成完整的覆盖率仪表板"""
        print("🚀 开始生成覆盖率仪表板...")

        # 加载历史数据
        history = self.load_coverage_history()

        if not history:
            print("❌ 没有找到覆盖率历史数据")
            return

        print(f"📊 加载了 {len(history)} 条历史记录")

        # 生成HTML仪表板
        self.generate_html_dashboard(history)

        print(f"🎉 覆盖率仪表板已生成到: {self.dashboard_dir}")


def main():
    """主函数"""
    generator = CoverageDashboardGenerator()
    generator.generate_dashboard()


if __name__ == "__main__":
    main()
