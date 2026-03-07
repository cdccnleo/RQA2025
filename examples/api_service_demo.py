"""
APIService演示脚本
展示API网关服务的核心功能
"""

from datetime import datetime
from src.core import EventBus, ServiceContainer
from src.services import APIService, APIVersion, APIEndpoint


def demo_handler(method, path, headers, body, backend):
    """演示处理器"""
    return {
        "message": "Hello from demo endpoint",
        "method": method,
        "path": path,
        "backend": backend,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """主演示函数"""
    print("🚀 APIService演示开始")
    print("=" * 50)

    # 创建基础组件
    event_bus = EventBus()
    container = ServiceContainer()

    # 创建API服务
    api_service = APIService(event_bus, container, "DemoAPIService")

    print("1. 服务初始化")
    print(f"   - 服务名称: {api_service.name}")
    print(f"   - 服务状态: {api_service.get_status()}")
    print(f"   - 默认端点数量: {len(api_service.routes)}")
    print()

    # 启动服务
    print("2. 启动服务")
    api_service.start()
    print(f"   - 服务状态: {api_service.get_status()}")
    print(f"   - 端点数量: {len(api_service.routes)}")
    print()

    # 注册自定义端点
    print("3. 注册自定义端点")
    demo_endpoint = APIEndpoint(
        path="/demo",
        method="GET",
        handler=demo_handler,
        description="演示端点",
        tags=["demo"]
    )

    success = api_service.register_endpoint(demo_endpoint)
    print(f"   - 注册结果: {success}")
    print(f"   - 端点数量: {len(api_service.routes)}")
    print()

    # 测试路由请求
    print("4. 测试路由请求")
    result = api_service.route_request("GET", "/demo")
    print(f"   - 请求结果: {result['success']}")
    if result['success']:
        print(f"   - 响应数据: {result['data']}")
        print(f"   - 响应时间: {result['response_time']:.3f}s")
    print()

    # 测试健康检查端点
    print("5. 测试健康检查端点")
    health_result = api_service.route_request("GET", "/health")
    print(f"   - 健康检查: {health_result['success']}")
    if health_result['success']:
        print(f"   - 健康状态: {health_result['data']}")
    print()

    # 测试统计端点
    print("6. 测试统计端点")
    stats_result = api_service.route_request("GET", "/stats")
    print(f"   - 统计获取: {stats_result['success']}")
    if stats_result['success']:
        stats = stats_result['data']
        print(f"   - 总请求数: {stats['total_requests']}")
        print(f"   - 成功请求: {stats['success_requests']}")
        print(f"   - 错误请求: {stats['error_requests']}")
        print(f"   - 平均响应时间: {stats['avg_response_time']:.3f}s")
    print()

    # 测试限流功能
    print("7. 测试限流功能")
    api_service.set_rate_limit("GET:/demo", 2)  # 限制2个请求

    # 前两个请求应该成功
    for i in range(2):
        result = api_service.route_request("GET", "/demo")
        print(f"   - 请求 {i+1}: {'成功' if result['success'] else '失败'}")

    # 第三个请求应该被限流
    result = api_service.route_request("GET", "/demo")
    print(f"   - 请求 3: {'成功' if result['success'] else '被限流'}")
    if not result['success']:
        print(f"   - 限流原因: {result['error']}")
    print()

    # 测试认证功能
    print("8. 测试认证功能")
    # 创建需要认证的端点

    def auth_handler(method, path, headers, body, backend):
        return {"message": "Authenticated endpoint", "user": "admin"}

    auth_endpoint = APIEndpoint(
        path="/auth",
        method="GET",
        handler=auth_handler,
        auth_required=True,
        description="需要认证的端点"
    )

    api_service.register_endpoint(auth_endpoint)

    # 不带认证的请求应该失败
    result = api_service.route_request("GET", "/auth")
    print(f"   - 无认证请求: {'成功' if result['success'] else '失败'}")
    if not result['success']:
        print(f"   - 失败原因: {result['error']}")

    # 带认证的请求应该成功
    headers = {"X-API-Key": "default_key"}
    result = api_service.route_request("GET", "/auth", headers=headers)
    print(f"   - 有认证请求: {'成功' if result['success'] else '失败'}")
    if result['success']:
        print(f"   - 响应数据: {result['data']}")
    print()

    # 测试负载均衡
    print("9. 测试负载均衡")
    api_service.add_load_balancer("demo_service", ["backend1", "backend2", "backend3"])

    for i in range(3):
        result = api_service.route_request("GET", "/demo")
        if result['success']:
            print(f"   - 请求 {i+1} 后端: {result['data']['backend']}")
    print()

    # 获取API文档
    print("10. 获取API文档")
    docs = api_service.get_api_docs()
    print(f"   - 支持的版本: {list(docs.keys())}")

    v1_docs = api_service.get_api_docs(APIVersion.V1)
    print(f"   - V1版本端点数量: {len(v1_docs['endpoints'])}")

    for endpoint in v1_docs['endpoints']:
        print(f"     - {endpoint['method']} {endpoint['path']}: {endpoint['description']}")
    print()

    # 获取服务统计
    print("11. 获取服务统计")
    stats = api_service.get_stats()
    print(f"   - 总请求数: {stats['total_requests']}")
    print(f"   - 成功请求: {stats['success_requests']}")
    print(f"   - 错误请求: {stats['error_requests']}")
    print(f"   - 平均响应时间: {stats['avg_response_time']:.3f}s")
    print(f"   - 活跃路由: {stats['active_routes']}")
    print(f"   - 活跃版本: {stats['active_versions']}")
    print()

    # 健康检查
    print("12. 健康检查")
    health = api_service._health_check()
    print(f"   - 总路由数: {health['total_routes']}")
    print(f"   - 活跃版本数: {health['active_versions']}")
    print(f"   - 限流请求数: {health['rate_limited_requests']}")
    print(f"   - 总请求数: {health['total_requests']}")
    print(f"   - 平均响应时间: {health['avg_response_time']:.3f}s")
    print()

    # 停止服务
    print("13. 停止服务")
    api_service.stop()
    print(f"   - 服务状态: {api_service.get_status()}")
    print()

    print("✅ APIService演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
