#!/usr/bin/env python3
"""
修复Docker应用404错误

检查并修复Docker容器中历史数据采集API的404错误
"""

import subprocess
import time
import requests
import sys

def restart_containers():
    """重启Docker容器"""
    print("🔄 重启RQA2025 Docker容器...")

    try:
        # 停止容器
        print("停止容器...")
        subprocess.run(['docker-compose', '-f', 'docker-compose.prod.yml', 'down'],
                      check=True, capture_output=True)

        # 重新启动容器
        print("重新启动容器...")
        subprocess.run(['docker-compose', '-f', 'docker-compose.prod.yml', 'up', '-d'],
                      check=True, capture_output=True)

        print("✅ 容器重启完成")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 容器重启失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 重启过程中出错: {e}")
        return False

def wait_for_services():
    """等待服务启动"""
    print("⏳ 等待服务启动...")

    max_attempts = 30
    for i in range(max_attempts):
        try:
            # 检查nginx
            nginx_resp = requests.get('http://localhost/health', timeout=5)
            if nginx_resp.status_code != 200:
                continue

            # 检查应用
            app_resp = requests.get('http://localhost:8000/health', timeout=5)
            if app_resp.status_code == 200:
                print("✅ 服务启动完成")
                return True

        except:
            pass

        if i < max_attempts - 1:
            print(f"等待中... ({i+1}/{max_attempts})")
            time.sleep(2)

    print("❌ 服务启动超时")
    return False

def test_apis():
    """测试API端点"""
    print("🧪 测试API端点...")

    test_cases = [
        ('http://localhost/api/v1/monitoring/data-collection/health', '数据采集健康检查'),
        ('http://localhost/api/v1/monitoring/historical-collection/status', '历史数据采集状态'),
        ('http://localhost/api/v1/monitoring/historical-collection/scheduler/status', '调度器状态'),
    ]

    results = {}
    for url, description in test_cases:
        try:
            response = requests.get(url, timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {description}: {response.status_code}")
            results[url] = response.status_code == 200
        except Exception as e:
            print(f"❌ {description}: 连接失败 - {e}")
            results[url] = False

    return results

def check_container_logs():
    """检查容器日志"""
    print("📋 检查应用容器日志...")

    try:
        result = subprocess.run(['docker', 'logs', '--tail', '20', 'rqa2025-app'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("应用容器最近日志:")
            # 只显示相关的错误日志
            lines = result.stdout.split('\n')
            relevant_lines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'fail', 'historical', '404']):
                    relevant_lines.append(line)

            if relevant_lines:
                for line in relevant_lines[-5:]:  # 显示最后5行相关日志
                    print(f"  {line}")
            else:
                print("  未发现明显的错误日志")
        else:
            print(f"获取应用日志失败: {result.stderr}")

    except Exception as e:
        print(f"检查应用日志失败: {e}")

def main():
    """主函数"""
    print("🔧 修复RQA2025 Docker应用404错误")
    print("=" * 50)

    # 重启容器
    if not restart_containers():
        print("❌ 容器重启失败，请手动检查Docker状态")
        return 1

    # 等待服务启动
    if not wait_for_services():
        print("❌ 服务启动失败")
        check_container_logs()
        return 1

    # 测试API
    time.sleep(3)  # 额外等待
    results = test_apis()

    # 检查结果
    historical_apis_working = all([
        results.get('http://localhost/api/v1/monitoring/historical-collection/status', False),
        results.get('http://localhost/api/v1/monitoring/historical-collection/scheduler/status', False)
    ])

    if historical_apis_working:
        print("\n🎉 修复成功！历史数据采集API现在正常工作")
        print("📱 现在可以访问监控页面: http://localhost/data-collection-monitor.html")
        return 0
    else:
        print("\n❌ 修复失败，历史数据采集API仍然不可用")
        check_container_logs()

        print("\n🔍 可能的解决方案:")
        print("1. 检查应用容器中的模块导入")
        print("2. 查看完整应用日志: docker logs rqa2025-app")
        print("3. 检查环境变量和配置文件")
        print("4. 确认所有依赖都正确安装")

        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n脚本退出码: {exit_code}")
    sys.exit(exit_code)