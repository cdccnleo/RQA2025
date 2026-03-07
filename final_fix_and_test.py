#!/usr/bin/env python3
"""
最终修复和测试数据源配置Method Not Allowed错误
"""

def final_fix_and_test():
    """最终修复和全面测试"""

    print("🔧 最终修复数据源配置Method Not Allowed错误")
    print("=" * 60)

    # 1. 验证修复内容
    print("\n1. 已实施的修复验证:")

    fixes = [
        ("添加OPTIONS路由", check_options_route()),
        ("修复nginx CORS配置", check_nginx_cors()),
        ("移除图表模拟数据", check_no_mock_data()),
        ("前端环境自适应", check_frontend_adaptive()),
    ]

    all_fixed = True
    for fix_name, is_fixed in fixes:
        status = '✅' if is_fixed else '❌'
        print(f"   {status} {fix_name}")
        if not is_fixed:
            all_fixed = False

    if not all_fixed:
        print("\n❌ 部分修复未完成，请检查上述项目")
        return False

    # 2. 端到端测试
    print("\n2. 端到端API测试:")
    test_results = test_end_to_end_api()

    if not test_results['success']:
        print(f"\n❌ API测试失败: {test_results['error']}")
        return False

    print("   ✅ API端点全部正常")

    # 3. 浏览器模拟测试
    print("\n3. 浏览器请求模拟测试:")
    browser_test = test_browser_simulation()

    if not browser_test['success']:
        print(f"\n❌ 浏览器模拟测试失败: {browser_test['error']}")
        return False

    print("   ✅ 浏览器请求模拟成功")

    # 4. 生成使用指南
    print("\n4. 使用指南:")
    print("   📋 生产环境:")
    print("      1. 启动系统: docker-compose up")
    print("      2. 访问: http://localhost:8080/data-sources")
    print("      3. 添加数据源应该正常工作")
    print("")
    print("   🛠️  开发环境:")
    print("      1. 启动API: uvicorn src.gateway.web.api:app --reload --port 8000")
    print("      2. 直接打开HTML文件或使用开发服务器")
    print("      3. 前端会自动检测环境并使用正确API地址")
    print("")
    print("   🔍 调试方法:")
    print("      1. F12打开开发者工具 → Network标签")
    print("      2. 尝试添加数据源，观察网络请求")
    print("      3. 检查OPTIONS和POST请求的状态码")
    print("      4. 查看控制台错误信息")

    print("\n🎉 所有修复已完成并验证通过！")
    print("   数据源配置现在应该可以正常添加数据源了")

    return True

def check_options_route():
    """检查OPTIONS路由是否存在"""
    try:
        with open('src/gateway/web/api.py', 'r', encoding='utf-8') as f:
            content = f.read()
        return '@app.options("/api/v1/data/sources")' in content
    except:
        return False

def check_nginx_cors():
    """检查nginx CORS配置"""
    try:
        with open('web-static/nginx.conf', 'r', encoding='utf-8') as f:
            content = f.read()
        # 检查OPTIONS请求的CORS头设置
        return ('if ($request_method = \'OPTIONS\')' in content and
                content.count('add_header \'Access-Control-Allow-Origin\' \'*\' always;') >= 2)
    except:
        return False

def check_no_mock_data():
    """检查是否移除了模拟数据"""
    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否还有硬编码的非零数据（排除所有0的数据）
        mock_patterns = [
            '[25, 22, 28, 24, 26, 23, 27]',
            '[18, 21, 19, 22, 20, 23, 19]',
            '[0, 0, 0, 0, 1200, 0, 0, 678',
            '[1200,',  # 任何包含1200的数据
            '[678,',   # 任何包含678的数据
        ]

        has_mock_data = False
        for pattern in mock_patterns:
            if pattern in content:
                has_mock_data = True
                break

        return not has_mock_data
    except:
        return False

def check_frontend_adaptive():
    """检查前端环境自适应"""
    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return ('window.location.protocol === \'file:\'' in content and
                'http://localhost:8000/api/v1/data/sources' in content)
    except:
        return False

def test_end_to_end_api():
    """端到端API测试"""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path.cwd() / 'src'))

        from gateway.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # 测试完整流程
        tests = [
            ('OPTIONS /api/v1/data/sources', lambda: client.options('/api/v1/data/sources')),
            ('GET /api/v1/data/sources', lambda: client.get('/api/v1/data/sources')),
            ('POST /api/v1/data/sources', lambda: client.post('/api/v1/data/sources', json={
                'id': 'final-test-123',
                'name': '最终测试数据源',
                'type': '股票数据',
                'url': 'https://test.final.com',
                'enabled': True
            })),
        ]

        for test_name, test_func in tests:
            response = test_func()
            if response.status_code not in [200, 201, 204]:
                return {'success': False, 'error': f'{test_name} 返回 {response.status_code}'}

        return {'success': True, 'error': None}

    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_browser_simulation():
    """浏览器请求模拟测试"""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path.cwd() / 'src'))

        from gateway.web.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # 模拟浏览器发送的请求头
        headers = {
            'Content-Type': 'application/json',
            'Origin': 'http://localhost:8080',
            'Referer': 'http://localhost:8080/data-sources',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }

        # 测试OPTIONS预检
        options_resp = client.options('/api/v1/data/sources', headers=headers)
        if options_resp.status_code != 200:
            return {'success': False, 'error': f'OPTIONS预检失败: {options_resp.status_code}'}

        # 测试POST请求
        test_data = {
            'id': 'browser-sim-123',
            'name': '浏览器模拟测试',
            'type': '股票数据',
            'url': 'https://browser.test.com',
            'enabled': True
        }

        post_resp = client.post('/api/v1/data/sources', json=test_data, headers=headers)
        if post_resp.status_code != 200:
            return {'success': False, 'error': f'POST请求失败: {post_resp.status_code}'}

        return {'success': True, 'error': None}

    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')

    success = final_fix_and_test()

    if success:
        print("\n" + "="*60)
        print("🎯 修复成功！数据源配置现在应该可以正常工作了")
        print("   如果仍有问题，请按上述调试步骤排查")
    else:
        print("\n" + "="*60)
        print("❌ 修复未完成，请检查错误信息并重新运行")
