#!/usr/bin/env python3
"""
特征工程任务命名规范验证脚本

用途：
1. 验证数据库中任务ID命名是否符合规范
2. 检查异常命名记录
3. 生成验证报告

规范要求：
- 所有特征提取任务必须以"feature_task_"为前缀
- 不允许使用"task_"前缀（不带feature_）

作者：AI Assistant
日期：2026-03-22
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate_task_naming() -> Tuple[bool, List[Dict], Dict]:
    """
    验证任务命名规范
    
    Returns:
        (is_valid, abnormal_records, stats)
    """
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    except ImportError as e:
        print(f"❌ 无法导入PostgreSQL模块: {e}")
        return False, [], {}
    
    conn = get_db_connection()
    if not conn:
        print("❌ 无法连接到PostgreSQL数据库")
        return False, [], {}
    
    try:
        cursor = conn.cursor()
        
        # 统计总记录数
        cursor.execute("SELECT COUNT(*) FROM feature_engineering_tasks")
        total_count = cursor.fetchone()[0]
        
        # 统计符合规范的记录数
        cursor.execute("""
            SELECT COUNT(*) 
            FROM feature_engineering_tasks 
            WHERE task_id LIKE 'feature_task_%'
        """)
        normal_count = cursor.fetchone()[0]
        
        # 查询异常记录（以task_开头但不是feature_task_开头）
        cursor.execute("""
            SELECT task_id, task_type, status, symbol, created_at
            FROM feature_engineering_tasks
            WHERE task_id LIKE 'task_%' 
              AND task_id NOT LIKE 'feature_task_%'
            ORDER BY created_at DESC
        """)
        
        abnormal_records = []
        for row in cursor.fetchall():
            abnormal_records.append({
                'task_id': row[0],
                'task_type': row[1],
                'status': row[2],
                'symbol': row[3],
                'created_at': row[4]
            })
        
        abnormal_count = len(abnormal_records)
        
        # 统计其他不符合规范的记录
        cursor.execute("""
            SELECT COUNT(*) 
            FROM feature_engineering_tasks 
            WHERE task_id NOT LIKE 'feature_task_%' 
              AND task_id NOT LIKE 'task_%'
        """)
        other_abnormal_count = cursor.fetchone()[0]
        
        cursor.close()
        
        stats = {
            'total_count': total_count,
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'other_abnormal_count': other_abnormal_count,
            'compliance_rate': (normal_count / total_count * 100) if total_count > 0 else 0
        }
        
        is_valid = abnormal_count == 0 and other_abnormal_count == 0
        
        return is_valid, abnormal_records, stats
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False, [], {}
    finally:
        return_db_connection(conn)


def fix_abnormal_naming() -> int:
    """
    修复异常命名记录
    
    Returns:
        修复的记录数
    """
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    except ImportError as e:
        print(f"❌ 无法导入PostgreSQL模块: {e}")
        return 0
    
    conn = get_db_connection()
    if not conn:
        print("❌ 无法连接到PostgreSQL数据库")
        return 0
    
    try:
        cursor = conn.cursor()
        
        # 修复以task_开头但不是feature_task_开头的记录
        cursor.execute("""
            UPDATE feature_engineering_tasks
            SET task_id = 'feature_' || task_id
            WHERE task_id LIKE 'task_%' 
              AND task_id NOT LIKE 'feature_task_%'
        """)
        
        fixed_count = cursor.rowcount
        conn.commit()
        cursor.close()
        
        return fixed_count
        
    except Exception as e:
        print(f"❌ 修复过程出错: {e}")
        conn.rollback()
        return 0
    finally:
        return_db_connection(conn)


def generate_report(is_valid: bool, abnormal_records: List[Dict], stats: Dict) -> str:
    """生成验证报告"""
    report = []
    report.append("=" * 80)
    report.append("特征工程任务命名规范验证报告")
    report.append("=" * 80)
    report.append(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"验证状态: {'✅ 通过' if is_valid else '❌ 未通过'}")
    report.append("")
    
    report.append("统计信息:")
    report.append(f"  总记录数: {stats.get('total_count', 0)}")
    report.append(f"  符合规范: {stats.get('normal_count', 0)}")
    report.append(f"  异常记录: {stats.get('abnormal_count', 0)}")
    report.append(f"  其他异常: {stats.get('other_abnormal_count', 0)}")
    report.append(f"  合规率: {stats.get('compliance_rate', 0):.2f}%")
    report.append("")
    
    if abnormal_records:
        report.append("异常记录列表:")
        report.append("-" * 80)
        for i, record in enumerate(abnormal_records[:20], 1):  # 最多显示20条
            report.append(f"{i}. {record['task_id']}")
            report.append(f"   类型: {record['task_type']}, 状态: {record['status']}, 股票: {record['symbol']}")
        
        if len(abnormal_records) > 20:
            report.append(f"... 还有 {len(abnormal_records) - 20} 条异常记录")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征工程任务命名规范验证工具')
    parser.add_argument('--fix', action='store_true', help='自动修复异常命名')
    parser.add_argument('--report', action='store_true', help='生成验证报告')
    
    args = parser.parse_args()
    
    print("🔍 开始验证任务命名规范...\n")
    
    # 执行验证
    is_valid, abnormal_records, stats = validate_task_naming()
    
    # 生成报告
    if args.report or not is_valid:
        report = generate_report(is_valid, abnormal_records, stats)
        print(report)
    
    # 自动修复
    if args.fix and not is_valid:
        print("\n🔧 开始修复异常命名...")
        fixed_count = fix_abnormal_naming()
        print(f"✅ 已修复 {fixed_count} 条异常记录")
        
        # 重新验证
        print("\n🔄 重新验证...")
        is_valid, abnormal_records, stats = validate_task_naming()
        report = generate_report(is_valid, abnormal_records, stats)
        print(report)
    
    # 返回状态码
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
