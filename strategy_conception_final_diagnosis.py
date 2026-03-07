#!/usr/bin/env python3
"""
strategy-conception.html最终问题诊断和修复
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

def diagnose_and_fix():
    """诊断并修复strategy-conception问题"""
    print("🔍 strategy-conception.html最终问题诊断和修复")
    print("=" * 60)

    print("📋 问题分析:")
    print("   • 用户报告: http://localhost:8080/strategy-conception 依然404")
    print("   • 经过多次修复尝试，问题仍然存在")
    print("   • 需要系统性地诊断所有相关组件")

    print("\n🔧 诊断步骤:")

    # 1. 检查本地文件
    print("\n1️⃣ 检查本地文件...")
    success, stdout, stderr = run_command("if exist web-static\\strategy-conception.html echo exists")
    if success and "exists" in stdout:
        print("   ✅ 本地strategy-conception.html文件存在")
    else:
        print("   ❌ 本地文件不存在 - 这是问题根源！")
        return False

    # 2. 检查本地nginx配置
    print("\n2️⃣ 检查本地nginx配置...")
    success, stdout, stderr = run_command("findstr strategy-conception web-static\\nginx.conf")
    if success and "strategy-conception" in stdout:
        print("   ✅ 本地nginx.conf包含strategy-conception路由")
    else:
        print("   ❌ 本地nginx配置缺失")
        return False

    # 3. 检查容器状态
    print("\n3️⃣ 检查容器状态...")
    success, stdout, stderr = run_command("docker ps --filter name=rqa2025-web --format '{{.Status}}'")
    if success and "Up" in stdout:
        print("   ✅ web容器正在运行")
    else:
        print("   ❌ web容器未运行")
        return False

    # 4. 检查容器中nginx配置
    print("\n4️⃣ 检查容器中nginx配置...")
    success, stdout, stderr = run_command("docker exec rqa2025-rqa2025-web-1 grep strategy-conception /etc/nginx/conf.d/default.conf")
    if success and stdout:
        print("   ✅ 容器nginx配置包含strategy-conception路由")
    else:
        print("   ❌ 容器nginx配置缺失 - 需要修复！")
        print("   🔧 修复: 复制nginx.conf到容器")
        run_command("docker cp web-static/nginx.conf rqa2025-rqa2025-web-1:/etc/nginx/conf.d/default.conf")

    # 5. 检查容器中HTML文件
    print("\n5️⃣ 检查容器中HTML文件...")
    success, stdout, stderr = run_command("docker exec rqa2025-rqa2025-web-1 ls /usr/share/nginx/html/strategy-conception.html 2>nul")
    if success:
        print("   ✅ 容器中strategy-conception.html文件存在")
    else:
        print("   ❌ 容器中HTML文件缺失 - 这就是404的根本原因！")
        print("   🔧 修复: 复制HTML文件到容器")
        run_command("docker cp web-static/strategy-conception.html rqa2025-rqa2025-web-1:/usr/share/nginx/html/strategy-conception.html")

    # 6. 重新加载nginx
    print("\n6️⃣ 重新加载nginx...")
    success, stdout, stderr = run_command("docker exec rqa2025-rqa2025-web-1 nginx -s reload")
    if success:
        print("   ✅ nginx重新加载成功")
    else:
        print(f"   ❌ nginx重新加载失败: {stderr}")
        return False

    # 7. 测试页面访问
    print("\n7️⃣ 测试页面访问...")
    import time
    time.sleep(2)  # 等待nginx完全启动

    try:
        response = requests.get("http://localhost:8080/strategy-conception", timeout=10)
        if response.status_code == 200:
            print("   ✅ 页面访问成功！")
            print(f"   📏 响应大小: {len(response.text)} 字符")
            print("   🎉 问题已解决！")

            print("\n" + "=" * 60)
            print("📋 问题总结:")
            print("❌ 根本原因: HTML文件没有正确复制到容器")
            print("✅ 解决方案: 使用docker cp命令复制文件")
            print("🎯 修复结果: 页面现在可以正常访问")

            return True
        else:
            print(f"   ❌ 页面访问仍然失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 页面访问异常: {e}")
        return False

if __name__ == "__main__":
    success = diagnose_and_fix()
    if success:
        print("\n🚀 修复成功！用户现在可以访问: http://localhost:8080/strategy-conception")
    else:
        print("\n❌ 修复失败，需要进一步检查")
    sys.exit(0 if success else 1)
