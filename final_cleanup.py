#!/usr/bin/env python3
"""
最终清理和验证财联社数据源
"""

import json
import os
import shutil

def cleanup_all_cls_sources():
    """清理所有配置文件中的财联社数据源"""
    print("🧹 最终清理财联社数据源")
    print("=" * 50)

    # 所有可能包含数据源配置的文件
    files_to_clean = [
        "data/data_sources_config.json",
        "src/data/data_sources_config.json",
        "data/production/data_sources_config.json",
        "data/testing/data_sources_config.json"
    ]

    total_cleaned = 0

    for file_path in files_to_clean:
        if os.path.exists(file_path):
            print(f"\n📄 处理文件: {file_path}")

            try:
                # 读取文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理不同格式
                if isinstance(data, list):
                    sources = data
                    format_type = "list"
                elif isinstance(data, dict):
                    sources = data.get('data_sources', [])
                    format_type = "dict"
                else:
                    print("  ❌ 未知文件格式，跳过")
                    continue

                original_count = len(sources)
                print(f"  原始数据源数量: {original_count} (格式: {format_type})")

                # 过滤掉财联社数据源
                cleaned_sources = []
                cls_removed = 0

                for source in sources:
                    if isinstance(source, dict) and source.get('name') == '财联社':
                        cls_removed += 1
                        print(f"  🗑️  删除财联社数据源 (ID: {repr(source.get('id'))})")
                    else:
                        cleaned_sources.append(source)

                # 保存清理后的数据
                if format_type == "list":
                    cleaned_data = cleaned_sources
                else:
                    data['data_sources'] = cleaned_sources
                    cleaned_data = data

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

                print(f"  ✅ 清理完成: {original_count} -> {len(cleaned_sources)} (删除 {cls_removed} 个财联社)")
                total_cleaned += cls_removed

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        else:
            print(f"\n📄 文件不存在: {file_path}")

    print(f"\n{'='*50}")
    print(f"🎉 清理完成！总共删除了 {total_cleaned} 个财联社数据源")
    return total_cleaned

def verify_cleanup():
    """验证清理结果"""
    print("\n🔍 验证清理结果")
    print("=" * 30)

    # 检查所有配置文件
    files_to_check = [
        "data/data_sources_config.json",
        "src/data/data_sources_config.json",
        "data/production/data_sources_config.json"
    ]

    all_clean = True

    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    sources = data
                elif isinstance(data, dict):
                    sources = data.get('data_sources', [])
                else:
                    sources = []

                cls_count = sum(1 for s in sources if isinstance(s, dict) and s.get('name') == '财联社')

                if cls_count > 0:
                    print(f"❌ {file_path}: 仍有 {cls_count} 个财联社数据源")
                    all_clean = False
                else:
                    print(f"✅ {file_path}: 清洁 ({len(sources)} 个数据源)")

            except Exception as e:
                print(f"❌ {file_path}: 验证失败 - {e}")
                all_clean = False
        else:
            print(f"⚪ {file_path}: 文件不存在")

    return all_clean

def main():
    """主函数"""
    print("🚀 财联社数据源最终清理程序")
    print("目标: 彻底删除所有配置文件中的财联社数据源")
    print()

    # 执行清理
    cleaned_count = cleanup_all_cls_sources()

    # 验证结果
    is_clean = verify_cleanup()

    print(f"\n{'='*50}")
    if is_clean:
        print("🎉 清理成功！所有财联社数据源已完全删除")
        print(f"📊 清理统计: 删除了 {cleaned_count} 个财联社数据源")
        print("✅ 数据源配置现在是清洁的")
        print("\n💡 建议:")
        print("   1. 重启所有相关服务")
        print("   2. 清除浏览器缓存")
        print("   3. 刷新数据源管理页面")
        return True
    else:
        print("❌ 清理不完整，仍有财联社数据源残留")
        print("🔧 请手动检查和清理相关文件")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
