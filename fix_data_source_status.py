#!/usr/bin/env python3
"""
数据源状态修复脚本
统一数据源状态格式，确保监控数据准确性
"""

import json
from datetime import datetime

def load_data_sources():
    """加载数据源配置"""
    config_file = "data/data_sources_config.json"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return []

def save_data_sources(sources):
    """保存数据源配置"""
    config_file = "data/data_sources_config.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)
        print(f"✅ 配置已保存到 {config_file}")
        return True
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        return False

def standardize_status(status):
    """标准化状态格式"""
    if not status:
        return "未测试"

    status = str(status).strip()

    # 连接成功的各种表示
    success_patterns = [
        "连接正常",
        "http 200 - 连接正常",
        "HTTP 200 - 连接正常",
        "连接成功"
    ]

    # 连接失败的各种表示
    failure_patterns = [
        "连接超时",
        "连接异常",
        "数据获取失败",
        "配置错误",
        "函数不存在"
    ]

    # 检查是否为成功状态
    for pattern in success_patterns:
        if pattern.lower() in status.lower():
            return "连接正常"

    # 检查是否为失败状态
    for pattern in failure_patterns:
        if pattern.lower() in status.lower():
            return status  # 保留具体的失败原因

    # 其他状态保持原样
    return status

def fix_data_source_status():
    """修复数据源状态"""
    print("🔧 数据源状态修复工具")
    print("=" * 50)

    # 加载配置
    sources = load_data_sources()
    if not sources:
        return False

    print(f"📊 加载了 {len(sources)} 个数据源配置")

    # 修复状态
    fixed_count = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, source in enumerate(sources):
        source_id = source.get("id", "")
        source_name = source.get("name", "")
        current_status = source.get("status", "")
        enabled = source.get("enabled", True)

        print(f"\n🔍 处理数据源: {source_name} ({source_id})")
        print(f"   当前状态: {current_status}")
        print(f"   启用状态: {enabled}")

        # 标准化状态
        original_status = current_status
        new_status = standardize_status(current_status)

        if new_status != original_status:
            print(f"   ✅ 状态修复: '{original_status}' -> '{new_status}'")
            source["status"] = new_status
            source["last_test"] = current_time  # 更新测试时间
            fixed_count += 1
        else:
            print(f"   ⏭️ 状态无需修复")

        # 对于禁用的数据源，设置特殊状态
        if not enabled:
            if source.get("status") != "已禁用":
                source["status"] = "已禁用"
                source["last_test"] = current_time
                print(f"   🔇 设置禁用状态")
                fixed_count += 1

    print(f"\n" + "=" * 50)
    print("📋 修复结果汇总")
    print("=" * 50)
    print(f"总数据源数量: {len(sources)}")
    print(f"修复的数据源: {fixed_count}")

    if fixed_count > 0:
        # 保存修复后的配置
        if save_data_sources(sources):
            print("✅ 数据源状态修复完成！")
            return True
        else:
            print("❌ 保存配置失败")
            return False
    else:
        print("✅ 所有数据源状态都已正确，无需修复")
        return True

def validate_fix():
    """验证修复结果"""
    print("\n🔍 验证修复结果...")
    sources = load_data_sources()

    status_counts = {}
    enabled_sources = 0
    disabled_sources = 0

    for source in sources:
        status = source.get("status", "未测试")
        enabled = source.get("enabled", True)

        if enabled:
            enabled_sources += 1
        else:
            disabled_sources += 1

        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1

    print(f"启用数据源: {enabled_sources}")
    print(f"禁用数据源: {disabled_sources}")
    print("状态分布:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}个")

    return True

if __name__ == "__main__":
    success = fix_data_source_status()
    if success:
        validate_fix()
    else:
        print("❌ 修复失败")
        exit(1)
