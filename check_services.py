#!/usr/bin/env python3
"""
RQA2025 服务状态检查脚本
检查后端API服务和前端静态文件服务是否正常运行
"""

import requests
import subprocess
import sys
import time

def check_service(name, url, timeout=5):
    """检查单个服务状态"""
    try:
        response = requests.get(url, timeout=timeout)
        return {
            'name': name,
            'url': url,
            'status': '运行中' if response.status_code == 200 else f'异常({response.status_code})',
            'code': response.status_code,
            'available': response.status_code == 200
        }
    except requests.exceptions.ConnectionError:
        return {
            'name': name,
            'url': url,
            'status': '未启动',
            'code': None,
            'available': False
        }
    except requests.exceptions.Timeout:
        return {
            'name': name,
            'url': url,
            'status': '响应超时',
            'code': None,
            'available': False
        }
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'status': f'错误: {str(e)}',
            'code': None,
            'available': False
        }

def check_processes():
    """检查相关进程"""
    processes = {}

    try:
        # 检查Python进程
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                              capture_output=True, text=True)
        python_processes = len([line for line in result.stdout.split('\n') if 'python.exe' in line])
        processes['python'] = python_processes

    except Exception as e:
        processes['python'] = f'检查失败: {e}'

    try:
        # 检查Node.js进程
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq node.exe'],
                              capture_output=True, text=True)
        node_processes = len([line for line in result.stdout.split('\n') if 'node.exe' in line])
        processes['node'] = node_processes

    except Exception as e:
        processes['node'] = f'检查失败: {e}'

    return processes

def check_ports():
    """检查端口占用"""
    ports = {}

    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)

        port_8000 = any(':8000' in line for line in result.stdout.split('\n'))
        port_8080 = any(':8080' in line for line in result.stdout.split('\n'))

        ports[8000] = port_8000
        ports[8080] = port_8080

    except Exception as e:
        ports['error'] = str(e)

    return ports

def main():
    """主函数"""
    print("🔍 RQA2025 服务状态检查")
    print("=" * 50)

    # 检查进程
    print("📊 进程状态:")
    processes = check_processes()
    print(f"   Python进程: {processes.get('python', '未知')}")
    print(f"   Node.js进程: {processes.get('node', '未知')}")

    # 检查端口
    print("\n🌐 端口占用:")
    ports = check_ports()
    if 'error' in ports:
        print(f"   端口检查失败: {ports['error']}")
    else:
        print(f"   端口8000 (后端API): {'✅ 已占用' if ports.get(8000) else '❌ 未占用'}")
        print(f"   端口8080 (前端服务): {'✅ 已占用' if ports.get(8080) else '❌ 未占用'}")

    # 检查服务
    print("\n🚀 服务状态:")
    services = [
        ('后端API服务', 'http://localhost:8000/health'),
        ('前端静态服务', 'http://localhost:8080/'),
        ('API数据源端点', 'http://localhost:8000/api/v1/data/sources'),
        ('前端主页', 'http://localhost:8080/index.html'),
        ('数据源配置页', 'http://localhost:8080/data-sources-config.html'),
    ]

    all_available = True
    for name, url in services:
        result = check_service(name, url)
        status_icon = '✅' if result['available'] else '❌'
        print(f"   {status_icon} {result['name']}: {result['status']}")

        if not result['available']:
            all_available = False

    # 总结
    print("\n" + "=" * 50)
    if all_available:
        print("🎉 所有服务运行正常！")
        print("\n📋 服务访问地址:")
        print("   🌐 前端主页: http://localhost:8080")
        print("   🔧 数据源配置: http://localhost:8080/data-sources-config.html")
        print("   📊 系统仪表板: http://localhost:8080/rqa2025-dashboard.html")
        print("   🔌 后端API: http://localhost:8000/api/v1/")
    else:
        print("⚠️  部分服务未正常运行")
        print("\n🔧 启动建议:")
        print("   启动后端API: python scripts/start_production.py")
        print("   启动前端服务: cd web-static && python -m http.server 8080")
        print("   检查端口冲突: netstat -ano | findstr :8000")
        print("   检查端口冲突: netstat -ano | findstr :8080")

    print("\n💡 提示:")
    print("   运行此脚本: python check_services.py")
    print("   查看详细日志: 检查终端输出")
    print("   重启服务: 先停止进程，再重新启动")

if __name__ == "__main__":
    main()
