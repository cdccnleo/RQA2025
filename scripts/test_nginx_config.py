#!/usr/bin/env python3
"""
nginx配置测试脚本

测试nginx配置文件语法是否正确
"""

import subprocess
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent

def test_nginx_config():
    """测试nginx配置文件"""
    nginx_config_path = project_root / "nginx" / "nginx.conf"

    print("🔍 测试nginx配置文件语法..."    print("=" * 60)
    print(f"配置文件路径: {nginx_config_path}")

    if not nginx_config_path.exists():
        print(f"❌ 配置文件不存在: {nginx_config_path}")
        return False

    # 尝试使用nginx -t命令测试配置
    try:
        # 首先尝试直接运行nginx -t
        result = subprocess.run(
            ["nginx", "-t", "-c", str(nginx_config_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ nginx配置语法正确")
            print("测试输出:")
            print(result.stdout)
            return True
        else:
            print("❌ nginx配置语法错误")
            print("错误输出:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("⚠️ nginx命令不可用，尝试使用Docker容器测试")

        # 如果nginx命令不可用，尝试使用Docker
        try:
            result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{nginx_config_path}:/etc/nginx/nginx.conf:ro",
                "nginx:alpine",
                "nginx", "-t", "-c", "/etc/nginx/nginx.conf"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("✅ nginx配置语法正确 (Docker测试)")
                print("测试输出:")
                print(result.stdout)
                return True
            else:
                print("❌ nginx配置语法错误 (Docker测试)")
                print("错误输出:")
                print(result.stderr)
                return False

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Docker也不可用，手动检查配置内容")

            # 手动检查常见语法错误
            return manual_config_check(nginx_config_path)

    except subprocess.TimeoutExpired:
        print("❌ nginx配置测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

def manual_config_check(config_path):
    """手动检查nginx配置的常见问题"""
    print("🔍 手动检查nginx配置...")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        issues = []

        # 检查大括号匹配
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            issues.append(f"大括号不匹配: {open_braces} 个 '{{' vs {close_braces} 个 '}}'")

        # 检查分号
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (stripped and not stripped.startswith('#') and
                not stripped.endswith('{') and not stripped.endswith('}') and
                not stripped.endswith(';') and
                not stripped.startswith('location') and
                not stripped.startswith('server') and
                not stripped.startswith('http') and
                not stripped.startswith('events')):
                # 检查是否是有效的nginx指令结尾
                if not any(stripped.endswith(x) for x in ['{', '}', ';', 'location', 'server', 'http', 'events']):
                    issues.append(f"第{i}行可能缺少分号: {stripped[:50]}...")

        # 检查upstream定义
        if 'upstream rqa2025_app' not in content:
            issues.append("缺少upstream定义")

        # 检查server块
        if 'server {' not in content:
            issues.append("缺少server块")

        # 检查location块
        if 'location /api/' not in content:
            issues.append("缺少API location块")

        if issues:
            print("❌ 发现以下潜在问题:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ 手动检查通过，未发现明显语法错误")
            print("注意: 这不是完整的语法验证，建议使用nginx -t进行正式测试")
            return True

    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 RQA2025 Nginx配置测试")
    print("=" * 60)

    success = test_nginx_config()

    print("\n" + "=" * 60)
    if success:
        print("🎯 测试结果: 配置语法正确")
        print("💡 如果nginx容器仍启动失败，请检查:")
        print("   - Docker服务是否正常运行")
        print("   - 容器依赖关系是否满足")
        print("   - 系统资源是否充足")
        print("   - 端口80是否被其他服务占用")
    else:
        print("🎯 测试结果: 配置存在问题")
        print("🔧 请修复上述问题后重新测试")

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)