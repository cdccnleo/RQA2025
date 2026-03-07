#!/usr/bin/env python3
"""
数据源连接状态验证脚本
检查数据源配置文件中的状态是否与实际连接测试结果一致
"""

import json
import asyncio
import aiohttp
import time
from datetime import datetime
import sys
import os

def load_data_sources():
    """直接加载数据源配置文件"""
    config_file = "data/data_sources_config.json"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return []

async def test_data_source_connection(source):
    """测试单个数据源的连接状态"""
    source_id = source.get("id", "")
    source_name = source.get("name", "")
    source_url = source.get("url", "")
    config = source.get("config", {})

    print(f"\n🔍 测试数据源: {source_name} ({source_id})")

    # 检查是否启用
    if not source.get("enabled", True):
        print(f"⏸️  数据源已禁用，跳过测试")
        return {
            "source_id": source_id,
            "name": source_name,
            "enabled": False,
            "current_status": source.get("status", "未测试"),
            "expected_status": "已禁用",
            "is_consistent": True,
            "test_result": "跳过"
        }

    # 对于AKShare数据源，使用特殊测试逻辑
    if "akshare" in source_id.lower() or source_id == "akshare_news_wallstreet":
        return await test_akshare_data_source(source)

    # 对于普通HTTP数据源
    return await test_http_data_source(source)

async def test_akshare_data_source(source):
    """测试AKShare数据源"""
    source_id = source.get("id", "")
    source_name = source.get("name", "")
    config = source.get("config", {})
    current_status = source.get("status", "未测试")

    akshare_function = config.get("akshare_function", "")

    if not akshare_function:
        print(f"❌ 配置错误: 缺少akshare_function")
        return {
            "source_id": source_id,
            "name": source_name,
            "enabled": True,
            "current_status": current_status,
            "expected_status": "配置错误",
            "is_consistent": current_status == "配置错误",
            "test_result": "配置错误"
        }

    try:
        import akshare
        print(f"📡 调用AKShare函数: {akshare_function}")

        # 检查函数是否存在
        if not hasattr(akshare, akshare_function):
            print(f"❌ 函数不存在: {akshare_function}")
            return {
                "source_id": source_id,
                "name": source_name,
                "enabled": True,
                "current_status": current_status,
                "expected_status": "函数不存在",
                "is_consistent": current_status == "函数不存在",
                "test_result": "函数不存在"
            }

        # 调用函数进行测试
        akshare_func = getattr(akshare, akshare_function)

        start_time = time.time()
        try:
            if akshare_function == "news_economic_baidu":
                data = await asyncio.to_thread(akshare_func, date="20241107")
            else:
                data = await asyncio.to_thread(akshare_func)

            elapsed_time = time.time() - start_time

            if data is not None and not data.empty and len(data) > 0:
                expected_status = "连接正常"
                test_result = f"成功 ({len(data)}条数据, {elapsed_time:.2f}s)"
                print(f"✅ 测试成功: 获取{len(data)}条数据, 耗时{elapsed_time:.2f}秒")
            else:
                expected_status = "数据获取失败"
                test_result = f"无数据 ({elapsed_time:.2f}s)"
                print(f"⚠️ 获取成功但无数据, 耗时{elapsed_time:.2f}秒")

        except Exception as e:
            elapsed_time = time.time() - start_time
            expected_status = f"连接异常: {str(e)[:50]}"
            test_result = f"异常 ({elapsed_time:.2f}s): {str(e)[:50]}"
            print(f"❌ 测试异常: {str(e)}, 耗时{elapsed_time:.2f}秒")

    except ImportError:
        print("❌ AKShare库未安装")
        return {
            "source_id": source_id,
            "name": source_name,
            "enabled": True,
            "current_status": current_status,
            "expected_status": "AKShare未安装",
            "is_consistent": False,
            "test_result": "AKShare未安装"
        }

    is_consistent = current_status == expected_status

    return {
        "source_id": source_id,
        "name": source_name,
        "enabled": True,
        "current_status": current_status,
        "expected_status": expected_status,
        "is_consistent": is_consistent,
        "test_result": test_result,
        "response_time": elapsed_time if 'elapsed_time' in locals() else None
    }

async def test_http_data_source(source):
    """测试HTTP数据源"""
    source_id = source.get("id", "")
    source_name = source.get("name", "")
    source_url = source.get("url", "")
    current_status = source.get("status", "未测试")

    if not source_url:
        print(f"❌ 配置错误: 缺少URL")
        return {
            "source_id": source_id,
            "name": source_name,
            "enabled": True,
            "current_status": current_status,
            "expected_status": "配置错误",
            "is_consistent": current_status == "配置错误",
            "test_result": "缺少URL"
        }

    try:
        print(f"🌐 测试HTTP连接: {source_url}")
        start_time = time.time()

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(source_url) as response:
                elapsed_time = time.time() - start_time

                if response.status == 200:
                    expected_status = f"HTTP {response.status} - 连接正常"
                    test_result = f"成功 ({elapsed_time:.2f}s)"
                    print(f"✅ HTTP {response.status}, 耗时{elapsed_time:.2f}秒")
                else:
                    expected_status = f"HTTP {response.status} - 服务错误"
                    test_result = f"HTTP {response.status} ({elapsed_time:.2f}s)"
                    print(f"❌ HTTP {response.status}, 耗时{elapsed_time:.2f}秒")

    except asyncio.TimeoutError:
        print("⏰ 连接超时")
        expected_status = "连接超时"
        test_result = "超时"
        elapsed_time = 10.0
    except Exception as e:
        elapsed_time = time.time() - start_time
        expected_status = f"连接异常: {str(e)[:50]}"
        test_result = f"异常 ({elapsed_time:.2f}s): {str(e)[:50]}"
        print(f"❌ 连接异常: {str(e)}, 耗时{elapsed_time:.2f}秒")

    is_consistent = current_status == expected_status

    return {
        "source_id": source_id,
        "name": source_name,
        "enabled": True,
        "current_status": current_status,
        "expected_status": expected_status,
        "is_consistent": is_consistent,
        "test_result": test_result,
        "response_time": elapsed_time if 'elapsed_time' in locals() else None
    }

async def main():
    """主函数"""
    print("🚀 数据源连接状态验证工具")
    print("=" * 60)

    # 加载数据源配置
    sources = load_data_sources()
    print(f"📊 共加载 {len(sources)} 个数据源配置")

    # 测试结果
    results = []
    consistent_count = 0
    total_enabled = 0

    for source in sources:
        if source.get("enabled", True):
            total_enabled += 1
            result = await test_data_source_connection(source)
            results.append(result)
            if result["is_consistent"]:
                consistent_count += 1

    # 输出结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)

    print(f"总数据源数量: {len(sources)}")
    print(f"启用数据源数量: {total_enabled}")
    print(f"状态一致数量: {consistent_count}")
    print(f"状态不一致数量: {total_enabled - consistent_count}")
    print(".1f")

    # 显示不一致的数据源
    inconsistent_sources = [r for r in results if not r["is_consistent"]]

    if inconsistent_sources:
        print("\n❌ 状态不一致的数据源:")
        print("-" * 60)
        for result in inconsistent_sources:
            print(f"数据源: {result['name']} ({result['source_id']})")
            print(f"当前状态: {result['current_status']}")
            print(f"期望状态: {result['expected_status']}")
            print(f"测试结果: {result['test_result']}")
            print("-" * 50)

        # 建议修复
        print("\n🔧 建议修复:")
        print("1. 运行数据源测试更新状态")
        print("2. 检查网络连接稳定性")
        print("3. 验证AKShare函数配置正确性")
        print("4. 更新配置文件中的状态字段")
    else:
        print("\n✅ 所有数据源状态一致！")

    # 保存详细结果
    output_file = "data_source_validation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "validation_time": datetime.now().isoformat(),
            "total_sources": len(sources),
            "enabled_sources": total_enabled,
            "consistent_sources": consistent_count,
            "inconsistent_sources": len(inconsistent_sources),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
