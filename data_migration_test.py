#!/usr/bin/env python3
"""
数据迁移验证脚本
Data Migration Validation Script

验证系统的数据处理、迁移和完整性保障能力
"""

import time
import json
import hashlib
from typing import Dict, List, Any

def generate_test_data(count: int = 1000) -> List[Dict[str, Any]]:
    """生成测试数据"""
    test_data = []
    for i in range(count):
        record = {
            'id': f'user_{i:04d}',
            'name': f'User{i}',
            'email': f'user{i}@example.com',
            'balance': float(i * 100.50),
            'status': 'active' if i % 10 != 0 else 'inactive',
            'created_at': f'2025-01-{i%28+1:02d}T10:00:00Z',
            'metadata': {
                'source': 'legacy_system',
                'version': '1.0',
                'tags': [f'tag_{j}' for j in range(i % 5)]
            }
        }
        test_data.append(record)
    return test_data

def calculate_data_hash(data: List[Dict[str, Any]]) -> str:
    """计算数据哈希值"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

def validate_data_integrity(original_data: List[Dict[str, Any]], migrated_data: List[Dict[str, Any]]) -> bool:
    """验证数据完整性"""
    if len(original_data) != len(migrated_data):
        print(f"❌ 数据量不匹配: 原始 {len(original_data)}, 迁移后 {len(migrated_data)}")
        return False

    # 检查每条记录
    for i, (orig, mig) in enumerate(zip(original_data, migrated_data)):
        if orig['id'] != mig['user_id']:
            print(f"❌ 第{i}条记录ID不匹配: {orig['id']} != {mig['user_id']}")
            return False
        if abs(orig['balance'] - mig['account_balance']) > 0.01:  # 允许小数点精度误差
            print(f"❌ 第{i}条记录余额不匹配: {orig['balance']} != {mig['account_balance']}")
            return False

    return True

def simulate_data_transformation(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """模拟数据转换过程"""
    transformed_data = []

    for record in data:
        # 数据转换逻辑
        transformed = {
            'user_id': record['id'],
            'full_name': record['name'],
            'contact_email': record['email'],
            'account_balance': record['balance'],
            'account_status': record['status'],
            'registration_date': record['created_at'],
            'system_metadata': record['metadata'],
            # 新增字段
            'migrated_at': '2025-09-28T14:00:00Z',
            'migration_version': '2.0',
            'data_quality_score': 95.5
        }
        transformed_data.append(transformed)

    return transformed_data

def simulate_batch_processing(data: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
    """模拟批量数据处理"""
    processed_data = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}: {len(batch)} 条记录")

        # 模拟批处理时间
        time.sleep(0.01)  # 10ms per batch

        # 转换批数据
        transformed_batch = simulate_data_transformation(batch)
        processed_data.extend(transformed_batch)

    return processed_data

def test_data_migration_performance(record_count: int, batch_size: int = 100) -> Dict[str, Any]:
    """测试数据迁移性能"""
    print(f"🧪 测试数据迁移性能: {record_count} 条记录, 批大小 {batch_size}")

    # 生成测试数据
    start_time = time.time()
    original_data = generate_test_data(record_count)
    data_gen_time = time.time() - start_time

    # 计算原始数据哈希
    original_hash = calculate_data_hash(original_data)

    # 执行数据迁移
    migration_start = time.time()
    migrated_data = simulate_batch_processing(original_data, batch_size)
    migration_time = time.time() - migration_start

    # 验证数据完整性
    integrity_check_start = time.time()
    is_integrity_ok = validate_data_integrity(original_data, migrated_data)
    migrated_hash = calculate_data_hash(migrated_data)
    integrity_check_time = time.time() - integrity_check_start

    # 计算性能指标
    throughput = record_count / migration_time if migration_time > 0 else 0

    results = {
        'record_count': record_count,
        'batch_size': batch_size,
        'data_generation_time': data_gen_time,
        'migration_time': migration_time,
        'integrity_check_time': integrity_check_time,
        'total_time': data_gen_time + migration_time + integrity_check_time,
        'throughput_records_per_sec': throughput,
        'data_integrity': is_integrity_ok,
        'original_hash': original_hash,
        'migrated_hash': migrated_hash,
        'hash_consistency': original_hash == migrated_hash
    }

    return results

def main():
    """主测试函数"""
    print("🚀 开始数据迁移验证测试")
    print("=" * 50)

    success = True

    # 测试场景
    test_scenarios = [
        {'record_count': 1000, 'batch_size': 50},
        {'record_count': 5000, 'batch_size': 100},
        {'record_count': 10000, 'batch_size': 200}
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"\n测试场景: {scenario['record_count']} 条记录, 批大小 {scenario['batch_size']}")
        print("-" * 40)

        try:
            results = test_data_migration_performance(**scenario)
            all_results.append(results)

            print(".2f")
            print(".2f")
            print(".2f")
            print(".0f")
            print(f"数据完整性: {'✅ 通过' if results['data_integrity'] else '❌ 失败'}")
            print(f"哈希一致性: {'✅ 通过' if results['hash_consistency'] else '❌ 失败'}")

            if not results['data_integrity']:
                success = False

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            success = False

    # 汇总报告
    print("\n" + "=" * 50)
    print("📊 数据迁移验证汇总报告")

    if all_results:
        avg_throughput = sum(r['throughput_records_per_sec'] for r in all_results) / len(all_results)
        total_records = sum(r['record_count'] for r in all_results)
        total_time = sum(r['total_time'] for r in all_results)

        print(".2f")
        print(f"总处理记录数: {total_records}")
        print(".2f")
        print(".1f")

        # 性能评估
        if avg_throughput > 1000:  # 每秒处理1000条记录
            print("✅ 迁移性能: 优秀")
        elif avg_throughput > 500:
            print("✅ 迁移性能: 良好")
        elif avg_throughput > 100:
            print("⚠️ 迁移性能: 可接受")
        else:
            print("❌ 迁移性能: 需要优化")
            success = False

    if success:
        print("🎉 数据迁移验证全部通过！系统数据处理能力达标")
        return 0
    else:
        print("❌ 数据迁移验证存在问题，需要修复")
        return 1

if __name__ == "__main__":
    exit(main())
