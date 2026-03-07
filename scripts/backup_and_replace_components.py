#!/usr/bin/env python3
"""
组件文件备份和替换脚本

安全地备份原始组件文件，并创建指向重构版本的新文件

创建时间: 2025-11-03
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def backup_file(file_path: Path, backup_dir: Path) -> Path:
    """备份单个文件"""
    if not file_path.exists():
        return None
    
    # 创建备份目录
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # 复制文件
    shutil.copy2(file_path, backup_path)
    print(f"  ✅ 备份: {file_path.name} → {backup_path.name}")
    
    return backup_path


def create_redirect_file(file_path: Path, redirect_to: str, description: str):
    """创建重定向文件"""
    content = f'''#!/usr/bin/env python3
"""
{description} - 重定向到重构版本

本文件是向后兼容的重定向文件。
实际实现在重构版本中，提供更好的功能和性能。

迁移时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
重构版本: {redirect_to}
"""

# 从重构版本导入所有内容
from {redirect_to} import *

__all__ = []  # 将从重构版本自动继承
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ 创建重定向: {file_path.name}")


def backup_and_replace_container_components() -> Dict[str, any]:
    """备份和替换container组件"""
    print("\n🔧 处理Container组件...")
    
    container_dir = PROJECT_ROOT / 'src' / 'core' / 'container'
    backup_dir = PROJECT_ROOT / 'backups' / 'container' / datetime.now().strftime("%Y%m%d")
    
    files_to_process = [
        'container_components.py',
        'factory_components.py',
        'locator_components.py',
        'registry_components.py',
        'resolver_components.py',
    ]
    
    results = {
        'backed_up': [],
        'redirected': [],
        'errors': []
    }
    
    for filename in files_to_process:
        file_path = container_dir / filename
        
        try:
            # 备份
            backup_path = backup_file(file_path, backup_dir)
            if backup_path:
                results['backed_up'].append(str(backup_path))
            
            # 创建重定向
            create_redirect_file(
                file_path,
                'src.core.container.refactored_container_components',
                f'Container组件 - {filename}'
            )
            results['redirected'].append(str(file_path))
            
        except Exception as e:
            results['errors'].append(f"{filename}: {e}")
            print(f"  ❌ 错误: {filename} - {e}")
    
    return results


def backup_and_replace_middleware_components() -> Dict[str, any]:
    """备份和替换middleware组件"""
    print("\n🔧 处理Middleware组件...")
    
    middleware_dir = PROJECT_ROOT / 'src' / 'core' / 'integration' / 'middleware'
    backup_dir = PROJECT_ROOT / 'backups' / 'middleware' / datetime.now().strftime("%Y%m%d")
    
    files_to_process = [
        'bridge_components.py',
        'connector_components.py',
        'middleware_components.py',
    ]
    
    results = {
        'backed_up': [],
        'redirected': [],
        'errors': []
    }
    
    for filename in files_to_process:
        file_path = middleware_dir / filename
        
        try:
            # 备份
            backup_path = backup_file(file_path, backup_dir)
            if backup_path:
                results['backed_up'].append(str(backup_path))
            
            # 创建重定向
            create_redirect_file(
                file_path,
                'src.core.integration.middleware.refactored_middleware_components',
                f'Middleware组件 - {filename}'
            )
            results['redirected'].append(str(file_path))
            
        except Exception as e:
            results['errors'].append(f"{filename}: {e}")
            print(f"  ❌ 错误: {filename} - {e}")
    
    return results


def backup_and_replace_business_process_components() -> Dict[str, any]:
    """备份和替换business process组件"""
    print("\n🔧 处理Business Process组件...")
    
    bp_dir = PROJECT_ROOT / 'src' / 'core' / 'orchestration' / 'business_process'
    backup_dir = PROJECT_ROOT / 'backups' / 'business_process' / datetime.now().strftime("%Y%m%d")
    
    files_to_process = [
        'coordinator_components.py',
        'manager_components.py',
        'orchestrator_components.py',
        'process_components.py',
        'workflow_components.py',
    ]
    
    results = {
        'backed_up': [],
        'redirected': [],
        'errors': []
    }
    
    for filename in files_to_process:
        file_path = bp_dir / filename
        
        try:
            # 备份
            backup_path = backup_file(file_path, backup_dir)
            if backup_path:
                results['backed_up'].append(str(backup_path))
            
            # 创建重定向
            create_redirect_file(
                file_path,
                'src.core.orchestration.business_process.refactored_business_process_components',
                f'Business Process组件 - {filename}'
            )
            results['redirected'].append(str(file_path))
            
        except Exception as e:
            results['errors'].append(f"{filename}: {e}")
            print(f"  ❌ 错误: {filename} - {e}")
    
    return results


def generate_backup_report(all_results: List[Dict]) -> str:
    """生成备份报告"""
    report = []
    report.append("=" * 70)
    report.append("文件备份和替换报告")
    report.append("=" * 70)
    report.append(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    total_backed_up = 0
    total_redirected = 0
    total_errors = 0
    
    for category, results in all_results:
        report.append(f"\n{category}")
        report.append("-" * 70)
        report.append(f"备份文件数: {len(results['backed_up'])}")
        report.append(f"重定向文件数: {len(results['redirected'])}")
        report.append(f"错误数: {len(results['errors'])}")
        
        total_backed_up += len(results['backed_up'])
        total_redirected += len(results['redirected'])
        total_errors += len(results['errors'])
        
        if results['errors']:
            report.append("\n错误列表:")
            for error in results['errors']:
                report.append(f"  ❌ {error}")
    
    report.append("\n" + "=" * 70)
    report.append("总计")
    report.append("=" * 70)
    report.append(f"总备份文件数: {total_backed_up}")
    report.append(f"总重定向文件数: {total_redirected}")
    report.append(f"总错误数: {total_errors}")
    report.append("")
    
    if total_errors == 0:
        report.append("✅ 所有文件备份和替换成功！")
    else:
        report.append(f"⚠️  有 {total_errors} 个错误需要处理")
    
    report.append("")
    report.append("备份位置: backups/")
    report.append("  - backups/container/")
    report.append("  - backups/middleware/")
    report.append("  - backups/business_process/")
    report.append("")
    
    return "\n".join(report)


def main():
    """主函数"""
    print("🚀 开始备份和替换组件文件...")
    print("=" * 70)
    
    all_results = []
    
    # 处理Container组件
    container_results = backup_and_replace_container_components()
    all_results.append(("Container组件", container_results))
    
    # 处理Middleware组件
    middleware_results = backup_and_replace_middleware_components()
    all_results.append(("Middleware组件", middleware_results))
    
    # 处理Business Process组件
    bp_results = backup_and_replace_business_process_components()
    all_results.append(("Business Process组件", bp_results))
    
    # 生成报告
    report = generate_backup_report(all_results)
    print("\n" + report)
    
    # 保存报告
    report_file = PROJECT_ROOT / 'test_logs' / 'backup_and_replace_report.txt'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 报告已保存到: {report_file.relative_to(PROJECT_ROOT)}")
    
    # 总结
    total_files = sum(len(r[1]['backed_up']) for r in all_results)
    total_errors = sum(len(r[1]['errors']) for r in all_results)
    
    if total_errors == 0:
        print(f"\n✅ 成功备份和替换 {total_files} 个文件！")
    else:
        print(f"\n⚠️  备份了 {total_files} 个文件，但有 {total_errors} 个错误")


if __name__ == '__main__':
    # 安全提示
    print("⚠️  警告: 此脚本将修改源代码文件")
    print("✅ 所有原始文件将被备份到 backups/ 目录")
    print("🔄 原始文件将被替换为指向重构版本的重定向文件")
    print("")
    
    response = input("确定继续吗？(yes/no): ")
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("❌ 操作已取消")

