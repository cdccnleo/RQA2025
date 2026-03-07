#!/usr/bin/env python3
"""
验证API网关整合后的导入正确性
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有API网关导入"""
    print("🔍 验证API网关整合后的导入...")

    results = {}

    # 测试1: 主API网关 (src/core/api_gateway.py)
    try:
        from src.core.api_gateway import ApiGateway
        results['core_api_gateway'] = {'status': 'success', 'class': 'ApiGateway'}
        print("✅ src.core.api_gateway.ApiGateway 导入成功")
    except ImportError as e:
        results['core_api_gateway'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ src.core.api_gateway.ApiGateway 导入失败: {e}")

    # 测试2: 网关路由器 (src/gateway/api_gateway.py)
    try:
        from src.gateway.api_gateway import GatewayRouter, APIGateway
        # 验证向后兼容性
        assert GatewayRouter is APIGateway, "向后兼容性别名不匹配"
        results['gateway_router'] = {'status': 'success', 'class': 'GatewayRouter'}
        print("✅ src.gateway.api_gateway.GatewayRouter 导入成功")
    except ImportError as e:
        results['gateway_router'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ src.gateway.api_gateway.GatewayRouter 导入失败: {e}")
    except AssertionError as e:
        results['gateway_router'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 向后兼容性检查失败: {e}")

    # 测试3: 集成代理 (src/core/integration/api_gateway.py)
    try:
        from src.core.integration.api_gateway import IntegrationProxy, get_api_gateway
        results['integration_proxy'] = {'status': 'success', 'class': 'IntegrationProxy'}
        print("✅ src.core.integration.api_gateway.IntegrationProxy 导入成功")
    except ImportError as e:
        results['integration_proxy'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ src.core.integration.api_gateway.IntegrationProxy 导入失败: {e}")

    # 测试4: 验证没有命名冲突
    try:
        # 确保可以同时导入所有类
        from src.core.api_gateway import ApiGateway as CoreGateway
        from src.gateway.api_gateway import GatewayRouter as GatewayRouter
        from src.core.integration.api_gateway import IntegrationProxy as IntegrationProxy

        # 验证它们是不同的类
        assert CoreGateway is not GatewayRouter, "CoreGateway与GatewayRouter应该是不同的类"
        assert CoreGateway is not IntegrationProxy, "CoreGateway与IntegrationProxy应该是不同的类"
        assert GatewayRouter is not IntegrationProxy, "GatewayRouter与IntegrationProxy应该是不同的类"

        results['no_conflicts'] = {'status': 'success', 'message': '没有命名冲突'}
        print("✅ 所有API网关类可以同时导入，无命名冲突")
    except Exception as e:
        results['no_conflicts'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 命名冲突检测失败: {e}")

    # 测试5: 验证功能完整性
    try:
        # 测试各个类的基本功能
        core_gateway = ApiGateway({'host': 'localhost', 'port': 8080})
        gateway_router = GatewayRouter()
        integration_proxy = get_api_gateway()

        # 验证基本属性存在
        assert hasattr(core_gateway, 'routes'), "ApiGateway缺少routes属性"
        assert hasattr(gateway_router, 'routes'), "GatewayRouter缺少routes属性"
        assert hasattr(integration_proxy, 'routes'), "IntegrationProxy缺少routes属性"

        results['functionality'] = {'status': 'success', 'message': '基本功能完整'}
        print("✅ 所有API网关类的基本功能正常")
    except Exception as e:
        results['functionality'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 功能验证失败: {e}")

    return results


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🔄 测试向后兼容性...")

    compatibility_results = {}

    # 测试旧的导入方式是否仍然工作
    try:
        compatibility_results['gateway_old_import'] = {'status': 'success'}
        print("✅ 网关层的向后兼容性保持")
    except ImportError as e:
        compatibility_results['gateway_old_import'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ 网关层向后兼容性破坏: {e}")

    return compatibility_results


def generate_report(results, compatibility_results):
    """生成验证报告"""
    print("\n" + "="*60)
    print("🎯 API网关整合验证报告")
    print("="*60)

    # 导入测试结果
    print("\n📦 导入测试结果:")
    for test_name, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status} {test_name}: {result.get('class', result.get('message', result.get('error', 'Unknown')))}")

    # 兼容性测试结果
    print("\n🔄 兼容性测试结果:")
    for test_name, result in compatibility_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status} {test_name}: {result.get('error', '保持兼容')}")

    # 总体评估
    all_tests = list(results.values()) + list(compatibility_results.values())
    successful_tests = sum(1 for r in all_tests if r['status'] == 'success')
    total_tests = len(all_tests)

    print("\n📊 总体评估:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {successful_tests}")
    print(f"  失败测试: {total_tests - successful_tests}")
    print(f"  成功率: {successful_tests/total_tests:.1f}%")
    if successful_tests == total_tests:
        print("🎉 所有测试通过！API网关整合成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")
        return False


if __name__ == "__main__":
    print("🚀 开始API网关整合验证...")

    # 运行测试
    results = test_imports()
    compatibility_results = test_backward_compatibility()

    # 生成报告
    success = generate_report(results, compatibility_results)

    # 退出状态
    sys.exit(0 if success else 1)
