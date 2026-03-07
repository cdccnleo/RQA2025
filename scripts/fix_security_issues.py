#!/usr/bin/env python3
"""
安全问题修复脚本

修复安全评估中发现的问题，包括文件权限、环境变量等
"""

import os
import stat
import subprocess
import sys
from pathlib import Path


class SecurityFixer:
    """安全问题修复器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def fix_file_permissions(self):
        """修复文件权限问题"""
        print("🔒 修复文件权限...")

        # 需要修复权限的目录
        dirs_to_fix = [
            self.project_root / 'src',
            self.project_root / 'tests',
            self.project_root / 'config',
            self.project_root / 'scripts'
        ]

        for dir_path in dirs_to_fix:
            if dir_path.exists():
                try:
                    # 移除世界写权限
                    current_mode = dir_path.stat().st_mode
                    new_mode = current_mode & ~stat.S_IWOTH  # 移除世界写权限

                    dir_path.chmod(new_mode)
                    print(f"  ✅ 已修复权限: {dir_path}")

                except Exception as e:
                    print(f"  ❌ 修复权限失败 {dir_path}: {e}")

    def fix_env_file_permissions(self):
        """修复.env文件权限"""
        print("🔒 修复.env文件权限...")

        env_files = [
            self.project_root / '.env',
            self.project_root / '.env.local',
            self.project_root / '.env.development',
            self.project_root / '.env.production'
        ]

        for env_file in env_files:
            if env_file.exists():
                try:
                    # 设置为只有所有者可读写
                    env_file.chmod(0o600)
                    print(f"  ✅ 已修复权限: {env_file}")

                except Exception as e:
                    print(f"  ❌ 修复权限失败 {env_file}: {e}")

    def check_sensitive_env_vars(self):
        """检查敏感环境变量"""
        print("🔍 检查敏感环境变量...")

        sensitive_vars = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'API_KEY']
        found_sensitive = []

        for var_name in os.environ:
            if any(sensitive in var_name.upper() for sensitive in sensitive_vars):
                found_sensitive.append(var_name)

        if found_sensitive:
            print("  ⚠️  发现敏感环境变量:")
            for var in found_sensitive:
                print(f"    - {var}")
            print("  💡 建议: 确保这些变量不会记录到日志中")

        return found_sensitive

    def create_secure_env_template(self):
        """创建安全的.env模板"""
        print("📝 创建安全的.env模板...")

        env_template = """# 安全环境变量配置模板
# 重要: 请勿将真实敏感信息提交到版本控制系统

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rqa_database
DB_USER=rqa_user
DB_PASSWORD=CHANGE_THIS_SECURE_PASSWORD

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_THIS_SECURE_PASSWORD

# InfluxDB配置
INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
INFLUXDB_TOKEN=CHANGE_THIS_SECURE_TOKEN
INFLUXDB_ORG=rqa_org
INFLUXDB_BUCKET=rqa_bucket

# JWT配置
JWT_SECRET_KEY=CHANGE_THIS_VERY_LONG_SECURE_RANDOM_KEY_32_CHARS_MINIMUM
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# API配置
API_KEY=CHANGE_THIS_SECURE_API_KEY
API_SECRET=CHANGE_THIS_SECURE_API_SECRET

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# 安全配置
DEBUG=false
LOG_LEVEL=INFO
ENABLE_AUDIT_LOG=true
ENABLE_ENCRYPTION=true

# 重要安全提醒:
# 1. 不要将真实密码和密钥提交到版本控制
# 2. 使用强密码和随机生成的密钥
# 3. 定期轮换密钥
# 4. 在生产环境中使用环境变量而不是文件
"""

        template_path = self.project_root / '.env.template'
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(env_template)
            print(f"  ✅ 已创建.env模板: {template_path}")

            # 设置模板文件权限
            template_path.chmod(0o644)

        except Exception as e:
            print(f"  ❌ 创建.env模板失败: {e}")

    def create_gitignore_security_rules(self):
        """确保.gitignore包含安全规则"""
        print("🔒 检查.gitignore安全规则...")

        gitignore_path = self.project_root / '.gitignore'
        security_patterns = [
            '# 安全敏感文件',
            '.env',
            '.env.local',
            '.env.*.local',
            'secrets/',
            '*.key',
            '*.pem',
            '*.p12',
            '*.pfx',
            'config/secrets/',
            'logs/*.log',
            'security_assessment_report.json',
            'coverage.xml',
            '.coverage'
        ]

        try:
            existing_content = ""
            if gitignore_path.exists():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()

            missing_patterns = []
            for pattern in security_patterns:
                if pattern not in existing_content:
                    missing_patterns.append(pattern)

            if missing_patterns:
                print("  📝 添加缺失的安全规则到.gitignore...")
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n' + '\n'.join(missing_patterns) + '\n')

                print(f"  ✅ 已添加 {len(missing_patterns)} 个安全规则到.gitignore")
            else:
                print("  ✅ .gitignore安全规则已完整")

        except Exception as e:
            print(f"  ❌ 更新.gitignore失败: {e}")

    def run_security_scan(self):
        """运行最终安全扫描验证"""
        print("🔍 运行最终安全验证...")

        try:
            # 运行安全评估测试
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/security/test_security_assessment.py::TestSecurityAssessment::test_comprehensive_security_assessment',
                '-v', '--tb=no'
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("  ✅ 安全评估通过")
                return True
            else:
                print("  ❌ 安全评估仍有问题")
                print(result.stdout)
                return False

        except Exception as e:
            print(f"  ❌ 运行安全评估失败: {e}")
            return False

    def run_all_fixes(self):
        """运行所有安全修复"""
        print("🚀 开始安全问题修复...")
        print("="*50)

        # 执行修复步骤
        self.fix_file_permissions()
        self.fix_env_file_permissions()
        self.check_sensitive_env_vars()
        self.create_secure_env_template()
        self.create_gitignore_security_rules()

        print("\n🔍 验证修复效果...")
        success = self.run_security_scan()

        print("\n" + "="*50)
        if success:
            print("🎉 安全修复完成！所有安全问题已解决")
        else:
            print("⚠️  安全修复完成，但仍有一些问题需要手动处理")

        return success


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    fixer = SecurityFixer(project_root)
    success = fixer.run_all_fixes()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
