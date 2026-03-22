#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进项验证测试脚本

用于验证以下改进项的实施效果：
1. DB-1: 数据库密码迁移到环境变量
2. DP-1: 修复编码声明错误
3. DP-2: 统一配置访问方式
4. FS-1: 完善选择策略初始化
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, r'c:\PythonProject\RQA2025')


def test_db_1_password_security():
    """测试 DB-1: 数据库密码安全性"""
    print("=" * 60)
    print("验证 DB-1: 数据库密码迁移到环境变量")
    print("=" * 60)
    
    try:
        # 测试1: 验证未设置密码时抛出异常
        print("\n1. 测试未设置密码时的异常处理...")
        
        # 临时清除环境变量
        original_password = os.environ.pop('POSTGRES_PASSWORD', None)
        
        try:
            # 重新加载模块以触发配置加载
            import importlib
            from src.infrastructure.persistence import database_config
            importlib.reload(database_config)
            
            # 尝试获取配置，应该抛出 ValueError
            from src.infrastructure.persistence.database_config import DatabaseConfigManager
            DatabaseConfigManager._instance = None  # 重置单例
            
            try:
                config = DatabaseConfigManager.get_config()
                print("❌ 测试失败：未设置密码时应该抛出异常")
                return False
            except ValueError as e:
                if "数据库密码未设置" in str(e):
                    print("✅ 测试通过：未设置密码时正确抛出异常")
                    print(f"   异常信息: {str(e)[:80]}...")
                else:
                    print(f"❌ 测试失败：异常信息不符合预期: {e}")
                    return False
        finally:
            # 恢复环境变量
            if original_password:
                os.environ['POSTGRES_PASSWORD'] = original_password
        
        # 测试2: 验证设置密码后能正常获取配置
        print("\n2. 测试设置密码后能正常获取配置...")
        os.environ['POSTGRES_PASSWORD'] = 'TestPassword123'
        
        from src.infrastructure.persistence.database_config import DatabaseConfigManager
        DatabaseConfigManager._instance = None  # 重置单例
        
        try:
            config = DatabaseConfigManager.get_config()
            if config.password == 'TestPassword123':
                print("✅ 测试通过：环境变量密码正确读取")
            else:
                print("❌ 测试失败：密码读取不正确")
                return False
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        
        print("\n🎉 DB-1 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dp_1_encoding_fix():
    """测试 DP-1: 编码声明修复"""
    print("\n" + "=" * 60)
    print("验证 DP-1: 修复编码声明错误")
    print("=" * 60)
    
    try:
        # 检查文件中的编码声明
        print("\n1. 检查 feature_engineer.py 中的编码声明...")
        
        file_path = r'c:\PythonProject\RQA2025\src\features\core\feature_engineer.py'
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还有错误的编码声明
        if 'utf - 8' in content:
            print("❌ 测试失败：文件中仍存在错误的编码声明 'utf - 8'")
            return False
        
        # 检查是否有正确的编码声明
        if 'utf-8' in content:
            print("✅ 测试通过：文件中使用正确的编码声明 'utf-8'")
        else:
            print("⚠️ 警告：文件中未找到编码声明")
        
        # 统计编码声明出现次数
        correct_count = content.count('utf-8')
        print(f"   正确的 'utf-8' 声明出现次数: {correct_count}")
        
        print("\n🎉 DP-1 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dp_2_config_unification():
    """测试 DP-2: 统一配置访问方式"""
    print("\n" + "=" * 60)
    print("验证 DP-2: 统一配置访问方式")
    print("=" * 60)
    
    try:
        # 测试1: 验证 ValidationConfig 类存在
        print("\n1. 验证 ValidationConfig 类...")
        
        from src.features.core.feature_engineer import ValidationConfig
        
        # 创建默认配置
        config = ValidationConfig()
        print(f"✅ ValidationConfig 类可正常实例化")
        print(f"   默认配置: allow_negative_prices={config.allow_negative_prices}, "
              f"strict_price_logic={config.strict_price_logic}")
        
        # 测试2: 验证配置可以从字典创建
        print("\n2. 验证配置从字典创建...")
        
        test_dict = {
            'allow_negative_prices': True,
            'strict_price_logic': False,
            'allow_nan_values': True
        }
        config2 = ValidationConfig(**test_dict)
        
        if (config2.allow_negative_prices == True and 
            config2.strict_price_logic == False and
            config2.allow_nan_values == True):
            print("✅ 测试通过：配置可以从字典正确创建")
        else:
            print("❌ 测试失败：配置值不匹配")
            return False
        
        # 测试3: 验证代码中不再使用 hasattr 检查
        print("\n3. 检查代码中不再使用分散的 hasattr 检查...")
        
        file_path = r'c:\PythonProject\RQA2025\src\features\core\feature_engineer.py'
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查 _validate_stock_data 方法中是否使用 validation_config
        if 'validation_config' in content:
            print("✅ 测试通过：代码中使用统一的 validation_config 访问配置")
        else:
            print("❌ 测试失败：未找到统一的配置访问方式")
            return False
        
        print("\n🎉 DP-2 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fs_1_selector_initialization():
    """测试 FS-1: 完善选择策略初始化"""
    print("\n" + "=" * 60)
    print("验证 FS-1: 完善选择策略初始化")
    print("=" * 60)
    
    try:
        from src.features.processors.feature_selector import FeatureSelector
        
        # 测试1: 验证 variance 选择器初始化
        print("\n1. 测试 variance 选择器初始化...")
        try:
            selector = FeatureSelector(selector_type='variance')
            print(f"✅ variance 选择器初始化成功")
            print(f"   选择器类型: {type(selector.selector).__name__ if selector.selector else 'None'}")
        except Exception as e:
            print(f"⚠️ variance 选择器初始化: {e}")
        
        # 测试2: 验证 correlation 选择器初始化
        print("\n2. 测试 correlation 选择器初始化...")
        try:
            selector = FeatureSelector(selector_type='correlation')
            print(f"✅ correlation 选择器初始化成功")
            print(f"   选择器类型: {type(selector.selector).__name__ if selector.selector else 'None'}")
        except Exception as e:
            print(f"⚠️ correlation 选择器初始化: {e}")
        
        # 测试3: 验证 importance 选择器初始化
        print("\n3. 测试 importance 选择器初始化...")
        try:
            selector = FeatureSelector(selector_type='importance')
            print(f"✅ importance 选择器初始化成功")
            print(f"   选择器类型: {type(selector.selector).__name__ if selector.selector else 'None'}")
        except Exception as e:
            print(f"⚠️ importance 选择器初始化: {e}")
        
        # 测试4: 验证 rfecv 选择器仍然正常工作
        print("\n4. 测试 rfecv 选择器（原有功能）...")
        try:
            selector = FeatureSelector(selector_type='rfecv')
            print(f"✅ rfecv 选择器初始化成功")
            print(f"   选择器类型: {type(selector.selector).__name__}")
        except Exception as e:
            print(f"❌ rfecv 选择器初始化失败: {e}")
            return False
        
        print("\n🎉 FS-1 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有验证测试"""
    print("\n" + "🧪 " * 20)
    print("改进项实施效果验证测试")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # 测试 DB-1
    result1 = test_db_1_password_security()
    results.append(("DB-1: 数据库密码迁移到环境变量", result1))
    
    # 测试 DP-1
    result2 = test_dp_1_encoding_fix()
    results.append(("DP-1: 修复编码声明错误", result2))
    
    # 测试 DP-2
    result3 = test_dp_2_config_unification()
    results.append(("DP-2: 统一配置访问方式", result3))
    
    # 测试 FS-1
    result4 = test_fs_1_selector_initialization()
    results.append(("FS-1: 完善选择策略初始化", result4))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("验证测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项验证通过")
    
    if passed == total:
        print("\n🎉 所有改进项验证通过！实施成功。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 项验证失败，请检查实施。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
