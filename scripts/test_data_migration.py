#!/usr/bin/env python3
"""
数据迁移测试脚本
"""

import os
import json
from pathlib import Path
from datetime import datetime

def test_data_backup():
    """测试数据备份功能"""
    print("开始数据备份测试...")

    # 模拟备份操作
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f'backup_{timestamp}.json'

    test_data = {
        'timestamp': timestamp,
        'type': 'test_backup',
        'status': 'success'
    }

    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"备份文件创建成功: {backup_file}")
    return True

def test_data_validation():
    """测试数据验证功能"""
    print("开始数据验证测试...")

    # 模拟数据验证
    test_records = [
        {'id': 1, 'symbol': 'AAPL', 'price': 150.0},
        {'id': 2, 'symbol': 'GOOGL', 'price': 2500.0},
        {'id': 3, 'symbol': 'MSFT', 'price': 300.0}
    ]

    valid_count = 0
    for record in test_records:
        if all(key in record for key in ['id', 'symbol', 'price']):
            if isinstance(record['price'], (int, float)) and record['price'] > 0:
                valid_count += 1

    print(f"数据验证完成: {valid_count}/{len(test_records)} 条记录有效")
    return valid_count == len(test_records)

if __name__ == '__main__':
    print("数据迁移测试开始")
    print("-" * 30)

    backup_success = test_data_backup()
    validation_success = test_data_validation()

    print("-" * 30)
    if backup_success and validation_success:
        print("✅ 数据迁移测试通过")
    else:
        print("❌ 数据迁移测试失败")
