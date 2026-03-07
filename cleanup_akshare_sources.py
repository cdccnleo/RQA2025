#!/usr/bin/env python3
"""
AKShare数据源清理和修复脚本
删除接口不存在和SSL证书验证失败的数据源，修复JSON解析错误的数据源
"""

import json
import sys
from typing import Dict, Any, List

def load_config() -> List[Dict[str, Any]]:
    """加载数据源配置文件"""
    try:
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        sys.exit(1)

def save_config(config: List[Dict[str, Any]]) -> None:
    """保存数据源配置文件"""
    try:
        with open('data/data_sources_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"✅ 配置文件已保存，共{len(config)}个数据源")
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        sys.exit(1)

def filter_akshare_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """筛选出AKShare数据源"""
    return [source for source in sources if source.get('url', '').startswith('https://akshare')]

def get_sources_to_remove() -> List[str]:
    """获取需要删除的数据源ID列表

    根据测试报告，需要删除以下数据源：
    1. 接口不存在：akshare_fund (fund_em_open_fund_daily不存在)
    2. SSL证书验证失败的另类数据源（如果存在）
    3. 暂时保留财经新闻数据源，稍后修复
    """
    return [
        'akshare_fund',  # 接口不存在
        # 另类数据源（如果存在的话）
        'akshare_alternative_weibo',
        'akshare_alternative_baidu_search',
        'akshare_alternative_wechat',
        'akshare_alternative_movie_boxoffice',
        'akshare_alternative_taobao_sales',
        'akshare_alternative_baltic_dry_index',
        'akshare_alternative_container_index',
        'akshare_alternative_air_quality',
        'akshare_alternative_weather',
    ]

def remove_problematic_sources(config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """删除有问题的AKShare数据源"""
    sources_to_remove = get_sources_to_remove()
    akshare_sources = filter_akshare_sources(config)

    print("🔍 分析当前AKShare数据源...")
    print(f"   总AKShare数据源数量: {len(akshare_sources)}")

    # 统计要删除的数据源
    existing_to_remove = [source for source in akshare_sources if source.get('id') in sources_to_remove]
    print(f"   需要删除的数据源: {len(existing_to_remove)}个")

    for source in existing_to_remove:
        print(f"   - {source.get('name', 'unknown')} ({source.get('id')})")

    # 删除数据源
    cleaned_config = []
    removed_count = 0

    for source in config:
        source_id = source.get('id', '')
        if source_id in sources_to_remove:
            print(f"🗑️  删除数据源: {source.get('name', 'unknown')} ({source_id})")
            removed_count += 1
        else:
            cleaned_config.append(source)

    print(f"✅ 已删除 {removed_count} 个有问题的数据源")
    return cleaned_config

def fix_news_sources(config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修复财经新闻数据源的JSON解析问题

    根据AKShare最新接口，财经新闻接口可能已经变更，
    使用更稳定的接口来替换有问题的新闻源
    """
    print("\n🔧 修复财经新闻数据源...")

    # 需要修复的新闻数据源ID
    news_sources_to_fix = [
        'akshare_news_js',        # 金十新闻 - JSON解析错误
        'akshare_news_sina',      # 新浪财经新闻 - JSON解析错误
        'akshare_news_wallstreet', # 华尔街见闻 - JSON解析错误
        'akshare_news_eastmoney', # 东方财富新闻 - JSON解析错误
        'akshare_news_all'        # 全量新闻数据 - JSON解析错误
    ]

    fixed_count = 0
    for i, source in enumerate(config):
        source_id = source.get('id', '')
        if source_id in news_sources_to_fix:
            print(f"🔧 修复数据源: {source.get('name', 'unknown')} ({source_id})")

            # 使用AKShare实际可用的新闻接口替换
            if source_id == 'akshare_news_js':
                # 金十新闻 -> 改用东方财富新闻接口
                config[i]['config']['news_source'] = 'stock_news_em'
                config[i]['config']['akshare_function'] = 'stock_news_em'
                config[i]['config']['description'] = 'AKShare东方财富财经新闻数据采集接口，实时财经资讯'
                config[i]['last_test'] = None
                config[i]['status'] = '未测试'

            elif source_id == 'akshare_news_sina':
                # 新浪财经新闻 -> 改用百度经济新闻接口
                config[i]['config']['news_source'] = 'news_economic_baidu'
                config[i]['config']['akshare_function'] = 'news_economic_baidu'
                config[i]['config']['description'] = 'AKShare百度财经新闻数据采集接口，包含宏观经济新闻'
                config[i]['last_test'] = None
                config[i]['status'] = '未测试'

            elif source_id == 'akshare_news_wallstreet':
                # 华尔街见闻 -> 改用财新新闻接口
                config[i]['config']['news_source'] = 'stock_news_main_cx'
                config[i]['config']['akshare_function'] = 'stock_news_main_cx'
                config[i]['config']['description'] = 'AKShare财新财经新闻数据采集接口，权威财经资讯'
                config[i]['last_test'] = None
                config[i]['status'] = '未测试'

            elif source_id == 'akshare_news_eastmoney':
                # 东方财富新闻 -> 保留使用stock_news_em
                config[i]['config']['news_source'] = 'eastmoney'
                config[i]['config']['akshare_function'] = 'stock_news_em'
                config[i]['config']['description'] = 'AKShare东方财富财经新闻数据采集接口，实时财经资讯'
                config[i]['last_test'] = None
                config[i]['status'] = '未测试'

            elif source_id == 'akshare_news_all':
                # 全量新闻数据 -> 使用综合新闻接口
                config[i]['config']['news_source'] = 'multiple_sources'
                config[i]['config']['akshare_function'] = 'news_economic_baidu'
                config[i]['config']['description'] = 'AKShare多源财经新闻数据采集接口，支持百度经济新闻等多个新闻源'
                config[i]['last_test'] = None
                config[i]['status'] = '未测试'

            fixed_count += 1

    print(f"✅ 已修复 {fixed_count} 个财经新闻数据源")
    return config

def validate_config(config: List[Dict[str, Any]]) -> bool:
    """验证配置文件"""
    print("\n🔍 验证配置文件...")

    akshare_sources = filter_akshare_sources(config)
    print(f"   剩余AKShare数据源: {len(akshare_sources)}个")

    # 检查是否有重复ID
    ids = [source.get('id', '') for source in config if source.get('id')]
    duplicates = set([x for x in ids if ids.count(x) > 1])
    if duplicates:
        print(f"❌ 发现重复ID: {list(duplicates)}")
        return False

    # 检查必填字段
    missing_fields = []
    for source in akshare_sources:
        if not source.get('id'):
            missing_fields.append(f"数据源缺少ID: {source.get('name', 'unknown')}")
        if not source.get('name'):
            missing_fields.append(f"数据源缺少名称: {source.get('id', 'unknown')}")
        if not source.get('type'):
            missing_fields.append(f"数据源缺少类型: {source.get('name', 'unknown')}")

    if missing_fields:
        print("❌ 发现配置问题:")
        for issue in missing_fields:
            print(f"   - {issue}")
        return False

    print("✅ 配置文件验证通过")
    return True

def main():
    """主函数"""
    print("🧹 AKShare数据源清理和修复工具")
    print("=" * 50)

    # 1. 加载配置
    config = load_config()
    original_count = len(config)
    print(f"📋 原始配置包含 {original_count} 个数据源")

    # 2. 删除有问题的AKShare数据源
    config = remove_problematic_sources(config)

    # 3. 修复财经新闻数据源
    config = fix_news_sources(config)

    # 4. 验证配置
    if validate_config(config):
        # 5. 保存配置
        save_config(config)
        final_count = len(config)
        removed_count = original_count - final_count
        print(f"\n🎉 清理完成！删除了 {removed_count} 个问题数据源")
    else:
        print("\n❌ 配置验证失败，请检查配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
