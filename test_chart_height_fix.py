#!/usr/bin/env python3
"""
测试数据源错误率和可用性统计chart高度修复
"""

def test_chart_height_fix():
    """测试chart高度修复"""
    print("🧪 测试数据源错误率和可用性统计chart高度修复")
    print("=" * 60)

    try:
        # 检查HTML文件中的canvas高度设置
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查错误率统计chart高度
        error_rate_height = 'errorRateChart" width="400" height="300"' in content
        availability_height = 'availabilityChart" width="400" height="300"' in content

        print("📊 Chart高度检查结果:")
        print(f"   ✅ 错误率统计chart高度: {'300px' if error_rate_height else '未修改'}")
        print(f"   ✅ 可用性统计chart高度: {'300px' if availability_height else '未修改'}")

        # 检查是否还有旧的高度设置
        old_error_height = 'errorRateChart" width="400" height="200"' in content
        old_availability_height = 'availabilityChart" width="400" height="200"' in content

        print("\n🔍 旧版本检查:")
        print(f"   ❌ 错误率统计旧高度(200px): {'仍存在' if old_error_height else '已清理'}")
        print(f"   ❌ 可用性统计旧高度(200px): {'仍存在' if old_availability_height else '已清理'}")

        if error_rate_height and availability_height:
            print("\n🎉 Chart高度修复成功！")
            print("修复内容:")
            print("   ✅ 数据源错误率统计chart高度: 200px → 300px")
            print("   ✅ 数据源可用性统计chart高度: 200px → 300px")
            print("   ✅ 充分利用下方空白区域，提升视觉效果")
            return True
        else:
            print("\n❌ Chart高度修复不完整")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_chart_height_fix()
    exit(0 if success else 1)
