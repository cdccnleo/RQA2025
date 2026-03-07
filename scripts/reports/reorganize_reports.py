#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 报告目录重新组织脚本
"""

import shutil
from datetime import datetime
from pathlib import Path


def create_new_structure():
    """创建新的目录结构"""
    reports_dir = Path("reports")

    # 新目录结构
    new_dirs = [
        "project/progress",
        "project/completion",
        "project/architecture",
        "project/deployment",
        "technical/testing",
        "technical/performance",
        "technical/security",
        "technical/quality",
        "technical/optimization",
        "business/analytics",
        "business/trading",
        "business/backtest",
        "business/compliance",
        "operational/monitoring",
        "operational/deployment",
        "operational/notification",
        "operational/maintenance",
        "research/ml_integration",
        "research/deep_learning",
        "research/reinforcement_learning",
        "research/continuous_optimization"
    ]

    for dir_path in new_dirs:
        full_path = reports_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")


def classify_file(filename):
    """分类文件"""
    filename_lower = filename.lower()

    # 项目报告
    if any(word in filename_lower for word in ["progress", "milestone", "status"]):
        return "project/progress"
    elif any(word in filename_lower for word in ["completion", "final", "complete"]):
        return "project/completion"
    elif any(word in filename_lower for word in ["architecture", "design", "code_review"]):
        return "project/architecture"
    elif any(word in filename_lower for word in ["deployment", "deploy", "install"]):
        return "project/deployment"

    # 技术报告
    elif any(word in filename_lower for word in ["test", "testing", "ast_analysis", "data_loader"]):
        return "technical/testing"
    elif any(word in filename_lower for word in ["performance", "benchmark", "optimization"]):
        return "technical/performance"
    elif any(word in filename_lower for word in ["security", "audit", "vulnerability"]):
        return "technical/security"
    elif any(word in filename_lower for word in ["quality", "code_quality"]):
        return "technical/quality"
    elif any(word in filename_lower for word in ["optimization", "improvement"]):
        return "technical/optimization"

    # 业务报告
    elif any(word in filename_lower for word in ["analytics", "analysis", "batch_report"]):
        return "business/analytics"
    elif any(word in filename_lower for word in ["trading", "trade", "strategy"]):
        return "business/trading"
    elif any(word in filename_lower for word in ["backtest", "backtesting", "consistency"]):
        return "business/backtest"
    elif any(word in filename_lower for word in ["compliance", "regulatory"]):
        return "business/compliance"

    # 运维报告
    elif any(word in filename_lower for word in ["monitoring", "monitor", "alert"]):
        return "operational/monitoring"
    elif any(word in filename_lower for word in ["deployment", "deploy", "blue_green"]):
        return "operational/deployment"
    elif any(word in filename_lower for word in ["notification", "notify"]):
        return "operational/notification"
    elif any(word in filename_lower for word in ["maintenance", "maintain"]):
        return "operational/maintenance"

    # 研究报告
    elif any(word in filename_lower for word in ["ml_integration", "machine_learning"]):
        return "research/ml_integration"
    elif any(word in filename_lower for word in ["deep_learning", "neural"]):
        return "research/deep_learning"
    elif any(word in filename_lower for word in ["reinforcement_learning", "rl_"]):
        return "research/reinforcement_learning"
    elif any(word in filename_lower for word in ["continuous_optimization", "auto_optimization"]):
        return "research/continuous_optimization"

    # 默认分类
    else:
        return "project/progress"


def move_files():
    """移动文件到新结构"""
    reports_dir = Path("reports")
    moved_count = 0

    # 遍历所有文件
    for file_path in reports_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
            # 跳过新创建的目录
            if file_path.parent.name in ["project", "technical", "business", "operational", "research"]:
                continue

            # 分类文件
            target_subdir = classify_file(file_path.name)
            target_path = reports_dir / target_subdir / file_path.name

            # 如果目标文件已存在，添加时间戳（仅在冲突时）
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                target_path = reports_dir / target_subdir / new_name

            # 移动文件
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
                print(f"📁 移动: {file_path.name} -> {target_subdir}/")
            except Exception as e:
                print(f"❌ 移动失败: {file_path.name} - {e}")

    print(f"✅ 共移动 {moved_count} 个文件")


def create_readme_files():
    """创建README文件"""
    reports_dir = Path("reports")

    category_names = {
        "project": "项目报告",
        "technical": "技术报告",
        "business": "业务报告",
        "operational": "运维报告",
        "research": "研究报告"
    }

    subcategory_names = {
        "progress": "进度报告",
        "completion": "完成报告",
        "architecture": "架构报告",
        "deployment": "部署报告",
        "testing": "测试报告",
        "performance": "性能报告",
        "security": "安全报告",
        "quality": "质量报告",
        "optimization": "优化报告",
        "analytics": "分析报告",
        "trading": "交易报告",
        "backtest": "回测报告",
        "compliance": "合规报告",
        "monitoring": "监控报告",
        "notification": "通知报告",
        "maintenance": "维护报告",
        "ml_integration": "机器学习集成",
        "deep_learning": "深度学习",
        "reinforcement_learning": "强化学习",
        "continuous_optimization": "持续优化"
    }

    for category_dir in reports_dir.iterdir():
        if category_dir.is_dir() and category_dir.name in category_names:
            for subcategory_dir in category_dir.iterdir():
                if subcategory_dir.is_dir():
                    readme_file = subcategory_dir / "README.md"
                    if not readme_file.exists():
                        content = f"""# {category_names[category_dir.name]} - {subcategory_names.get(subcategory_dir.name, subcategory_dir.name)}

## 📋 概述

本目录包含{category_names[category_dir.name]}中的{subcategory_names.get(subcategory_dir.name, subcategory_dir.name)}。

## 📁 文件列表

<!-- 文件列表将自动生成 -->

## 📊 统计信息

- 总文件数: 0
- 最后更新: {datetime.now().strftime('%Y-%m-%d')}

---

**维护者**: 项目团队  
**状态**: ✅ 活跃维护
"""
                        readme_file.write_text(content, encoding='utf-8')
                        print(f"✅ 创建README: {subcategory_dir}")


def main():
    """主函数"""
    print("🚀 开始重新组织reports目录...")

    # 1. 创建新结构
    create_new_structure()

    # 2. 移动文件
    move_files()

    # 3. 创建README文件
    create_readme_files()

    print("✅ 重新组织完成！")


if __name__ == "__main__":
    main()
