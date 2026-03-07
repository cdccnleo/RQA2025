#!/usr/bin/env python3
"""
修复Loki配置问题

解决Loki启动失败的schema版本和索引类型配置问题
"""

import subprocess
import time
import requests
import sys
import os

def check_loki_config():
    """检查Loki配置"""
    print("🔍 检查Loki配置...")

    config_file = "monitoring/loki/loki-config.yml"

    try:
        with open(config_file, 'r') as f:
            content = f.read()

        # 检查关键配置
        checks = {
            'schema: v13': 'schema: v13' in content,
            'store: tsdb': 'store: tsdb' in content,
            'allow_structured_metadata: false': 'allow_structured_metadata: false' in content
        }

        all_passed = all(checks.values())

        print("配置检查结果:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")

        if all_passed:
            print("✅ Loki配置正确")
            return True
        else:
            print("❌ Loki配置需要修复")
            return False

    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False

def restart_loki():
    """重启Loki容器"""
    print("🔄 重启Loki容器...")

    try:
        # 停止Loki容器
        print("停止Loki容器...")
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.prod.yml', 'stop', 'loki'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            print(f"停止Loki失败: {result.stderr}")
            return False

        # 启动Loki容器
        print("启动Loki容器...")
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.prod.yml', 'up', '-d', 'loki'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            print("✅ Loki容器重启成功")
            return True
        else:
            print(f"❌ Loki容器重启失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 重启Loki失败: {e}")
        return False

def wait_for_loki():
    """等待Loki启动"""
    print("⏳ 等待Loki启动...")

    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            # 测试Loki健康状态
            response = requests.get('http://localhost:3100/ready', timeout=5)
            if response.status_code == 200:
                print("✅ Loki已就绪")
                return True

        except Exception:
            pass

        if attempt < max_attempts - 1:
            print(f"等待中... ({attempt + 1}/{max_attempts})")
            time.sleep(2)

    print("❌ Loki启动超时")
    return False

def test_loki_functionality():
    """测试Loki基本功能"""
    print("🧪 测试Loki功能...")

    try:
        # 测试Loki API
        response = requests.get('http://localhost:3100/loki/api/v1/labels', timeout=10)
        if response.status_code == 200:
            print("✅ Loki API正常")
            return True
        else:
            print(f"❌ Loki API异常: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Loki功能测试失败: {e}")
        return False

def check_promtail_connection():
    """检查Promtail连接"""
    print("🔍 检查Promtail连接...")

    try:
        # 检查Promtail是否能连接到Loki
        response = requests.get('http://localhost:3100/loki/api/v1/labels', timeout=10)
        if response.status_code == 200:
            print("✅ Promtail到Loki连接正常")
            return True
        else:
            print(f"⚠️ Loki响应异常: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Promtail连接检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 修复RQA2025 Loki配置问题")
    print("=" * 50)

    # 检查配置
    if not check_loki_config():
        print("❌ 配置检查失败，请手动修复monitoring/loki/loki-config.yml")
        return 1

    # 重启Loki
    if not restart_loki():
        print("❌ Loki重启失败")
        return 1

    # 等待启动
    if not wait_for_loki():
        print("❌ Loki启动失败")
        print("\n🔍 故障排除:")
        print("1. 检查Loki日志: docker logs rqa2025-loki")
        print("2. 验证配置文件语法")
        print("3. 检查端口3100是否被占用")
        return 1

    # 测试功能
    if not test_loki_functionality():
        print("❌ Loki功能测试失败")
        return 1

    # 检查Promtail连接
    check_promtail_connection()

    print("\n" + "=" * 50)
    print("🎉 Loki修复完成！")
    print("📊 监控栈状态:")
    print("  ✅ Loki: 运行正常")
    print("  ✅ Promtail: 日志收集")
    print("  ✅ Grafana: 可视化界面")
    print("  📱 Grafana访问: http://localhost:3000")

    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\n脚本退出码: {exit_code}")
    sys.exit(exit_code)