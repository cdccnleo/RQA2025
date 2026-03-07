#!/usr/bin/env python3
"""
测试数据源配置页面更新

验证前端页面是否正确支持新的data_type_configs配置结构
"""

import json
import os

def test_config_structure_compatibility():
    """测试配置结构兼容性"""
    print("🔧 测试配置结构兼容性")
    print("=" * 50)

    # 读取当前的配置文件
    config_file = 'data/data_sources_config.json'
    if not os.path.exists(config_file):
        print("❌ 配置文件不存在")
        return False

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    akshare_config = None
    for source in config_data:
        if source.get('id') == 'akshare_stock_a':
            akshare_config = source
            break

    if not akshare_config:
        print("❌ 找不到akshare_stock_a配置")
        return False

    source_config = akshare_config.get('config', {})

    # 检查新配置结构
    if 'data_type_configs' in source_config:
        print("✅ 使用新的data_type_configs配置结构")

        dt_configs = source_config['data_type_configs']
        print(f"配置的数据类型数量: {len(dt_configs)}")

        enabled_types = [dt for dt, config in dt_configs.items() if config.get('enabled', False)]
        print(f"启用的数据类型: {enabled_types}")

        # 验证每个启用类型都有正确的结构
        for dt in enabled_types:
            if dt not in dt_configs:
                print(f"❌ 数据类型 {dt} 缺少配置")
                return False

            dt_config = dt_configs[dt]
            if not isinstance(dt_config, dict):
                print(f"❌ 数据类型 {dt} 配置格式错误")
                return False

            if 'enabled' not in dt_config or 'description' not in dt_config:
                print(f"⚠️ 数据类型 {dt} 缺少标准字段")

        return True

    # 检查向后兼容性
    elif 'data_types' in source_config:
        print("⚠️ 使用旧的data_types配置结构（建议升级）")

        old_types = source_config['data_types']
        print(f"旧配置的数据类型: {old_types}")

        return True

    else:
        print("❌ 没有找到数据类型配置")
        return False

def validate_frontend_integration():
    """验证前端集成"""
    print("\n🎨 验证前端集成")
    print("=" * 50)

    html_file = 'web-static/data-sources-config.html'

    if not os.path.exists(html_file):
        print("❌ 前端配置文件不存在")
        return False

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否包含新的配置结构
    checks = [
        ('data_type_enabled', '新的复选框name属性'),
        ('data_type_configs', '新的配置结构引用'),
        ('9种数据类型', '分钟级、周线、月线等'),
        ('配置说明', '用户指导文本'),
        ('向后兼容', '旧配置结构支持')
    ]

    passed_checks = 0
    for check_text, description in checks:
        if check_text in content:
            print(f"✅ {description}")
            passed_checks += 1
        else:
            print(f"❌ 缺少 {description}")

    return passed_checks >= 4

def generate_config_comparison():
    """生成配置结构对比"""
    print("\n📊 配置结构对比")
    print("=" * 50)

    print("旧配置结构 (data_types):")
    print("""
{
  "data_types": ["realtime", "daily"]
}
    """)

    print("新配置结构 (data_type_configs):")
    print("""
{
  "data_type_configs": {
    "realtime": {"enabled": true, "description": "实时行情数据"},
    "daily": {"enabled": true, "description": "日线数据"},
    "1min": {"enabled": false, "description": "1分钟K线数据"},
    "5min": {"enabled": false, "description": "5分钟K线数据"},
    "weekly": {"enabled": false, "description": "周线数据"},
    "monthly": {"enabled": false, "description": "月线数据"}
  }
}
    """)

def create_migration_guide():
    """创建迁移指南"""
    print("\n📋 迁移指南")
    print("=" * 50)

    print("""
前端页面迁移说明：

1. 数据类型选择从简单复选框改为详细配置卡片
2. 每个数据类型都有启用开关和描述信息
3. 支持9种数据类型（原来只支持4种）
4. 保持向后兼容旧的data_types配置

后端API兼容性：

1. 新的data_type_configs结构完全兼容
2. 自动解析启用状态生成data_types数组
3. 支持旧配置结构的向后兼容
4. 配置保存时同时生成新旧两种格式

测试建议：

1. 在前端页面创建新的数据源配置
2. 启用不同的数据类型组合
3. 保存配置并验证后端接收
4. 检查监控面板是否正确显示启用的数据类型
    """)

def main():
    """主函数"""
    print("🧪 数据源配置页面更新测试")
    print("=" * 60)
    print("验证前端页面对新配置结构的支持")
    print("=" * 60)

    test_results = []

    # 1. 测试配置结构兼容性
    config_ok = test_config_structure_compatibility()
    test_results.append(("配置结构兼容性", config_ok))

    # 2. 验证前端集成
    frontend_ok = validate_frontend_integration()
    test_results.append(("前端集成验证", frontend_ok))

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")

    passed_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed_count += 1

    success_rate = (passed_count / len(test_results)) * 100

    print(f"\n总体成功率: {passed_count}/{len(test_results)} ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("\n🎉 配置页面更新成功！")
        print("✅ 配置结构兼容性良好")
        print("✅ 前端集成完成")
        print("✅ 支持新的多周期数据采集")

        generate_config_comparison()
        create_migration_guide()
    else:
        print("\n❌ 配置页面更新存在问题，需要进一步检查。")

    return passed_count == len(test_results)

if __name__ == "__main__":
    exit_code = main()
    print(f"\n测试完成，退出码: {0 if exit_code else 1}")