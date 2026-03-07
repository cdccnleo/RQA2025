#!/usr/bin/env python3
"""
环境变量配置脚本

帮助用户设置生产环境所需的必要环境变量
"""

import os
from pathlib import Path
import getpass


class EnvironmentSetup:
    """环境变量设置器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.required_vars = {
            'JWT_SECRET_KEY': {
                'description': 'JWT令牌签名密钥',
                'sensitive': True,
                'default': 'your-jwt-secret-key-change-in-production'
            },
            'DATABASE_URL': {
                'description': '数据库连接URL',
                'sensitive': True,
                'default': 'postgresql://user:password@localhost:5432/rqa_prod'
            },
            'REDIS_HOST': {
                'description': 'Redis服务器主机',
                'sensitive': False,
                'default': 'localhost'
            },
            'REDIS_PORT': {
                'description': 'Redis服务器端口',
                'sensitive': False,
                'default': '6379'
            },
            'SECRET_KEY': {
                'description': '应用密钥',
                'sensitive': True,
                'default': 'your-secret-key-change-in-production'
            }
        }

    def check_current_environment(self):
        """检查当前环境变量状态"""
        print("🔍 检查当前环境变量状态...\n")

        status = {}
        for var_name, config in self.required_vars.items():
            is_set = var_name in os.environ
            value = os.environ.get(var_name, '未设置')
            masked_value = self._mask_sensitive_value(value, config['sensitive'])

            status[var_name] = {
                'is_set': is_set,
                'value': masked_value,
                'description': config['description']
            }

            status_icon = "✅" if is_set else "❌"
            print(f"{status_icon} {var_name}: {masked_value}")
            print(f"   说明: {config['description']}\n")

        return status

    def _mask_sensitive_value(self, value, is_sensitive):
        """掩码敏感信息"""
        if not is_sensitive:
            return value
        if len(value) <= 8:
            return '*' * len(value)
        return value[:4] + '*' * (len(value) - 8) + value[-4:]

    def interactive_setup(self):
        """交互式环境变量设置"""
        print("⚙️  环境变量配置向导")
        print("="*50)

        print("\n📋 需要配置以下环境变量:")
        for var_name, config in self.required_vars.items():
            print(f"• {var_name}: {config['description']}")

        print("\n🔧 配置选项:")
        print("1. 自动生成默认值")
        print("2. 手动输入自定义值")
        print("3. 跳过配置")

        while True:
            try:
                choice = input("\n请选择配置方式 (1/2/3): ").strip()

                if choice == '1':
                    self._auto_setup()
                    break
                elif choice == '2':
                    self._manual_setup()
                    break
                elif choice == '3':
                    print("⏭️  跳过环境变量配置")
                    break
                else:
                    print("❌ 无效选择，请输入1、2或3")

            except KeyboardInterrupt:
                print("\n\n🛑 配置已取消")
                return

    def _auto_setup(self):
        """自动设置默认值"""
        print("\n🔧 自动配置环境变量...")

        env_file = self.project_root / '.env'
        env_content = "# 生产环境变量配置\n# 请修改为实际的生产环境值\n\n"

        for var_name, config in self.required_vars.items():
            value = config['default']
            env_content += f"{var_name}={value}\n"
            print(f"✅ {var_name} = {self._mask_sensitive_value(value, config['sensitive'])}")

        # 写入.env文件
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print("\n📄 环境变量已保存到 .env 文件")
        print(f"📍 文件位置: {env_file}")
        print("\n⚠️  重要提醒:")
        print("• 请编辑 .env 文件，将默认值替换为实际的生产环境值")
        print("• 敏感信息请妥善保管，不要提交到版本控制系统")
        print("• 生产环境部署前，请确保所有变量都已正确配置")

    def _manual_setup(self):
        """手动设置自定义值"""
        print("\n🔧 手动配置环境变量...")

        env_vars = {}

        for var_name, config in self.required_vars.items():
            print(f"\n📝 配置 {var_name}")
            print(f"   说明: {config['description']}")

            while True:
                if config['sensitive']:
                    value = getpass.getpass(f"   请输入 {var_name} 的值: ")
                else:
                    value = input(f"   请输入 {var_name} 的值 (默认: {config['default']}): ").strip()

                if not value:
                    value = config['default']

                # 验证输入
                if self._validate_input(var_name, value):
                    env_vars[var_name] = value
                    masked_value = self._mask_sensitive_value(value, config['sensitive'])
                    print(f"   ✅ 已设置: {masked_value}")
                    break
                else:
                    print("   ❌ 输入无效，请重新输入")

        # 保存到.env文件
        self._save_env_file(env_vars)

    def _validate_input(self, var_name, value):
        """验证输入值"""
        if not value or value.strip() == '':
            return False

        # 特定变量的验证规则
        if var_name == 'REDIS_PORT':
            try:
                port = int(value)
                return 1 <= port <= 65535
            except ValueError:
                return False

        if var_name in ['DATABASE_URL', 'REDIS_HOST']:
            # 基本URL格式验证
            return len(value) > 5

        return True

    def _save_env_file(self, env_vars):
        """保存环境变量到文件"""
        env_file = self.project_root / '.env'

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write("# 生产环境变量配置\n")
            f.write(
                f"# 生成时间: {os.environ.get('USERNAME', 'unknown')} @ {os.path.basename(os.getcwd())}\n")
            f.write("# 请妥善保管此文件，不要提交到版本控制系统\n\n")

            for var_name, value in env_vars.items():
                f.write(f"{var_name}={value}\n")

        print("\n📄 环境变量已保存!")
        print(f"📍 文件位置: {env_file}")
        print(f"✅ 已配置 {len(env_vars)} 个环境变量")

    def load_env_file(self):
        """加载.env文件"""
        env_file = self.project_root / '.env'

        if env_file.exists():
            print("📂 发现 .env 文件，正在加载...")
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                            print(f"✅ 加载: {key}")

            print("📂 .env 文件加载完成\n")
        else:
            print("📂 未发现 .env 文件\n")

    def generate_env_template(self):
        """生成环境变量模板"""
        template_file = self.project_root / '.env.template'

        with open(template_file, 'w', encoding='utf-8') as f:
            f.write("# 生产环境变量配置模板\n")
            f.write("# 复制此文件为 .env 并填写实际值\n\n")

            for var_name, config in self.required_vars.items():
                f.write(f"# {config['description']}\n")
                f.write(f"{var_name}={config['default']}\n\n")

        print("📄 环境变量模板已生成!")
        print(f"📍 文件位置: {template_file}")
        print("\n📋 使用说明:")
        print("1. 复制 .env.template 为 .env")
        print("2. 编辑 .env 文件，填入实际值")
        print("3. 不要将 .env 文件提交到版本控制")

    def create_setup_summary(self):
        """创建设置总结"""
        print("\n" + "="*50)
        print("📊 环境变量配置总结")
        print("="*50)

        # 检查状态
        status = self.check_current_environment()

        total_vars = len(self.required_vars)
        set_vars = sum(1 for s in status.values() if s['is_set'])

        print(f"\n📈 配置统计:")
        print(f"   总变量数: {total_vars}")
        print(f"   已设置: {set_vars}")
        print(f"   未设置: {total_vars - set_vars}")
        print(".1f")
        if set_vars == total_vars:
            print("\n🎉 所有环境变量已正确配置!")
            print("✅ 可以开始生产部署流程")
            return True
        else:
            print("\n⚠️  还有环境变量需要配置")
            print("📋 请完成以下变量的配置:")
            for var_name, stat in status.items():
                if not stat['is_set']:
                    print(f"   • {var_name}: {stat['description']}")
            return False


def main():
    """主函数"""
    print("=== RQA2025环境变量配置工具 ===\n")

    setup = EnvironmentSetup()

    # 检查当前状态
    print("1️⃣ 检查当前环境变量状态")
    setup.check_current_environment()

    # 提供配置选项
    print("\n2️⃣ 环境变量配置选项")
    print("a) 生成环境变量模板")
    print("b) 自动配置默认值")
    print("c) 手动配置自定义值")
    print("d) 加载现有 .env 文件")
    print("q) 退出")

    while True:
        try:
            choice = input("\n请选择操作 (a/b/c/d/q): ").strip().lower()

            if choice == 'a':
                setup.generate_env_template()
            elif choice == 'b':
                setup._auto_setup()
            elif choice == 'c':
                setup._manual_setup()
            elif choice == 'd':
                setup.load_env_file()
            elif choice == 'q':
                break
            else:
                print("❌ 无效选择，请输入 a、b、c、d 或 q")

        except KeyboardInterrupt:
            print("\n\n🛑 配置已取消")
            break

    # 生成总结
    setup.create_setup_summary()

    print("\n🚀 下一步:")
    print("• 配置完成后，重新运行快速检查脚本")
    print("• 确认所有检查通过后再开始部署")
    print("• 安全保管 .env 文件，不要提交到版本控制")


if __name__ == "__main__":
    main()
