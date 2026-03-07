#!/usr/bin/env python3
"""
测试dashboard.html部署到容器
"""

import requests
import subprocess
import sys

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def test_dashboard_deployment():
    """测试dashboard部署"""
    print("🧪 测试dashboard.html部署到容器")
    print("=" * 60)

    issues = []

    # 1. 检查本地文件
    print("📁 检查本地文件...")
    success, stdout, stderr = run_command("if exist web-static\\dashboard.html echo exists")
    if success and "exists" in stdout:
        print("   ✅ 本地dashboard.html文件存在")
    else:
        print("   ❌ 本地dashboard.html文件不存在")
        issues.append("本地文件缺失")

    # 2. 检查容器状态
    print("\n🐳 检查容器状态...")
    success, stdout, stderr = run_command("docker ps --filter name=rqa2025-web --format '{{.Status}}'")
    if success and "Up" in stdout:
        print("   ✅ web容器正在运行")
    else:
        print("   ❌ web容器未运行")
        issues.append("容器未运行")

    # 3. 检查容器中文件
    print("\n📄 检查容器中文件...")
    success, stdout, stderr = run_command("docker exec rqa2025-rqa2025-web-1 ls -la /usr/share/nginx/html/dashboard.html")
    if success and "dashboard.html" in stdout:
        print("   ✅ 容器中dashboard.html文件存在")
        # 提取文件大小
        size_part = stdout.split()[-5] if len(stdout.split()) > 5 else "unknown"
        print(f"   📏 文件大小: {size_part} bytes")
    else:
        print("   ❌ 容器中dashboard.html文件不存在")
        issues.append("容器文件缺失")

    # 4. 检查nginx配置
    print("\n⚙️ 检查nginx配置...")
    success, stdout, stderr = run_command("docker exec rqa2025-rqa2025-web-1 grep -A 2 'location /dashboard' /etc/nginx/conf.d/default.conf")
    if success and "dashboard.html" in stdout and "rqa2025-dashboard.html" not in stdout:
        print("   ✅ nginx配置正确指向dashboard.html")
    else:
        print("   ❌ nginx配置不正确")
        print(f"   配置内容: {stdout}")
        issues.append("nginx配置错误")

    # 5. 测试页面访问
    print("\n🌐 测试页面访问...")
    try:
        response = requests.get("http://localhost:8080/dashboard", timeout=10)

        if response.status_code == 200:
            print("   ✅ /dashboard路由返回200 OK")
            print(f"   📏 响应大小: {len(response.text)} 字符")

            # 检查HTML内容
            if "<!DOCTYPE html>" in response.text and "RQA2025 量化交易系统" in response.text:
                print("   ✅ HTML内容正确")
            else:
                print("   ⚠️ HTML内容可能不完整")
                issues.append("HTML内容异常")
        else:
            print(f"   ❌ /dashboard路由返回{response.status_code}")
            issues.append(f"HTTP状态码错误: {response.status_code}")

    except Exception as e:
        print(f"   ❌ 页面访问失败: {e}")
        issues.append(f"访问失败: {e}")

    # 6. 总结
    print("\n" + "=" * 60)
    if issues:
        print("❌ 发现问题:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n🔧 需要修复上述问题")
        return False
    else:
        print("🎉 dashboard.html部署成功！")
        print("   ✅ 本地文件存在")
        print("   ✅ 容器正在运行")
        print("   ✅ 文件已挂载到容器")
        print("   ✅ nginx配置正确")
        print("   ✅ 页面可正常访问")
        print("\n🚀 用户现在可以通过 http://localhost:8080/dashboard 访问dashboard页面")
        return True

if __name__ == "__main__":
    success = test_dashboard_deployment()
    sys.exit(0 if success else 1)
