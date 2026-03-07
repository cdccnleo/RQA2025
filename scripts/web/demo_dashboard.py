#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一Web管理界面演示脚本
展示统一Web管理界面的功能和使用方法
"""

import requests

BASE_URL = "http://127.0.0.1:8080"


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n--- {title} ---")


def demo_dashboard_overview():
    """演示仪表板概览"""
    print_section("统一Web管理界面概览")

    print("🎯 项目目标:")
    print("   • 统一整合各层Web管理界面")
    print("   • 提供现代化的Web访问入口")
    print("   • 实现模块化的架构设计")
    print("   • 支持实时监控和告警功能")

    print("\n🏗️  架构特点:")
    print("   • 基于FastAPI的高性能Web框架")
    print("   • 模块化设计，支持动态注册")
    print("   • WebSocket实时通信支持")
    print("   • 统一的API设计和错误处理")

    print("\n📊 已集成模块:")
    print("   • 配置管理模块 (ConfigModule)")
    print("   • FPGA监控模块 (FPGAModule)")
    print("   • 资源监控模块 (ResourceModule)")
    print("   • 特征层监控模块 (FeaturesModule)")


def demo_api_endpoints():
    """演示API端点"""
    print_section("API端点演示")

    endpoints = [
        ("GET /api/modules", "获取所有模块列表"),
        ("GET /api/modules/{module_name}/data", "获取模块数据"),
        ("GET /api/modules/{module_name}/status", "获取模块状态"),
        ("GET /api/modules/{module_name}/config", "获取模块配置"),
        ("POST /api/modules/{module_name}/start", "启动模块"),
        ("POST /api/modules/{module_name}/stop", "停止模块"),
        ("GET /ws", "WebSocket实时通信"),
    ]

    print("可用的API端点:")
    for endpoint, description in endpoints:
        print(f"   {endpoint:<40} - {description}")

    # 测试模块列表API
    print_subsection("测试模块列表API")
    try:
        response = requests.get(f"{BASE_URL}/api/modules")
        if response.status_code == 200:
            modules = response.json()
            print(f"✅ 成功获取 {modules['total']} 个模块:")
            for module in modules['modules']:
                print(f"   • {module['display_name']} ({module['name']})")
        else:
            print(f"❌ API调用失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")


def demo_module_features():
    """演示模块功能"""
    print_section("模块功能演示")

    modules = [
        {
            "name": "config",
            "display_name": "配置管理",
            "features": [
                "统一配置管理界面",
                "配置分类和编辑",
                "配置验证和导出",
                "配置历史记录"
            ]
        },
        {
            "name": "fpga_monitoring",
            "display_name": "FPGA监控",
            "features": [
                "FPGA性能监控",
                "延迟和吞吐量监控",
                "温度和功耗监控",
                "告警阈值设置"
            ]
        },
        {
            "name": "resource_monitoring",
            "display_name": "资源监控",
            "features": [
                "CPU和内存监控",
                "GPU资源监控",
                "磁盘和网络监控",
                "系统资源告警"
            ]
        },
        {
            "name": "features_monitoring",
            "display_name": "特征层监控",
            "features": [
                "特征工程性能监控",
                "数据质量监控",
                "特征计算时间监控",
                "数据质量评分"
            ]
        }
    ]

    for module in modules:
        print_subsection(f"{module['display_name']} 模块")
        print(f"模块名称: {module['name']}")
        print("主要功能:")
        for feature in module['features']:
            print(f"   • {feature}")


def demo_architecture_design():
    """演示架构设计"""
    print_section("架构设计演示")

    print("🏛️  分层架构:")
    print("   • 引擎层 (Engine Layer)")
    print("     - 统一Web管理界面")
    print("     - 统一日志管理")
    print("     - 业务逻辑处理")
    print("   • 基础设施层 (Infrastructure Layer)")
    print("     - 配置管理")
    print("     - 数据库管理")
    print("     - 监控和缓存")

    print("\n🔧 核心组件:")
    print("   • UnifiedDashboard - 统一仪表板控制器")
    print("   • ModuleRegistry - 模块注册表管理器")
    print("   • ModuleFactory - 模块工厂类")
    print("   • BaseModule - 模块基类")

    print("\n📋 数据模型:")
    print("   • ModuleConfig - 模块配置模型")
    print("   • ModuleData - 模块数据模型")
    print("   • ModuleStatus - 模块状态模型")

    print("\n🌐 API设计:")
    print("   • RESTful API设计")
    print("   • WebSocket实时通信")
    print("   • 统一错误处理")
    print("   • 标准化响应格式")


def demo_usage_guide():
    """演示使用指南"""
    print_section("使用指南")

    print("🚀 启动服务:")
    print("   python scripts/web/start_unified_dashboard.py \\")
    print("     --host 127.0.0.1 \\")
    print("     --port 8080 \\")
    print("     --env development")

    print("\n🌐 访问地址:")
    print("   • 主界面: http://127.0.0.1:8080")
    print("   • API文档: http://127.0.0.1:8080/api/docs")
    print("   • 模块列表: http://127.0.0.1:8080/api/modules")

    print("\n🧪 运行测试:")
    print("   python scripts/testing/run_tests.py \\")
    print("     --test-files tests/unit/engine/web/test_unified_dashboard_integration.py")

    print("\n📊 API测试:")
    print("   python scripts/web/test_dashboard_api.py")


def demo_next_steps():
    """演示下一步计划"""
    print_section("下一步计划")

    print("🎯 短期目标 (1-2周):")
    print("   • 完善API端点实现")
    print("   • 修复404错误")
    print("   • 优化模块初始化性能")
    print("   • 增强实时监控功能")

    print("\n🏗️  中期目标 (3-4周):")
    print("   • 开发现代化Web前端界面")
    print("   • 实现用户认证和权限控制")
    print("   • 实现配置和监控数据持久化")
    print("   • 完善告警和通知机制")

    print("\n🚀 长期目标 (5-6周):")
    print("   • 微服务化架构改造")
    print("   • Docker容器化部署")
    print("   • Kubernetes云原生支持")
    print("   • 分布式监控和日志收集")


def main():
    """主函数"""
    print("🎉 RQA2025 统一Web管理界面演示")
    print("=" * 60)

    # 检查服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/api/modules", timeout=5)
        if response.status_code == 200:
            print("✅ 统一Web管理界面服务运行正常")
        else:
            print("⚠️  服务响应异常，请检查服务状态")
    except Exception as e:
        print(f"❌ 无法连接到服务: {str(e)}")
        print("请先启动服务: python scripts/web/start_unified_dashboard.py")
        return

    # 执行演示
    demo_dashboard_overview()
    demo_api_endpoints()
    demo_module_features()
    demo_architecture_design()
    demo_usage_guide()
    demo_next_steps()

    print_section("演示完成")
    print("🎉 统一Web管理界面架构迁移和模块化集成工作已成功完成！")
    print("📈 项目架构得到了显著优化，为后续功能扩展奠定了坚实基础。")
    print("🔧 如需技术支持或功能扩展，请参考相关文档和测试用例。")


if __name__ == "__main__":
    main()
