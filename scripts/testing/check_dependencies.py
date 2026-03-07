#!/usr/bin/env python3
"""
RQA2025 AI覆盖率自动化依赖检查脚本
检查所有必要的外部依赖和系统要求
"""

import os
import sys
import asyncio
import aiohttp
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DependencyChecker:
    """依赖检查器"""

    def __init__(self):
        self.project_root = project_root
        self.check_results = {}

    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🐍 检查Python版本...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
            print("   需要Python 3.8或更高版本")
            return False

        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True

    def check_conda_environment(self) -> bool:
        """检查conda环境"""
        print("🔧 检查conda环境...")

        if 'rqa' not in sys.prefix and 'rqa' not in sys.executable:
            print("❌ 未在conda rqa环境中运行")
            print("   请运行: conda activate rqa")
            return False

        print("✅ conda rqa环境已激活")
        return True

    def check_python_packages(self) -> bool:
        """检查Python包依赖"""
        print("📦 检查Python包依赖...")

        required_packages = {
            'aiohttp': '异步HTTP客户端',
            'pytest': '测试框架',
            'pytest-cov': '覆盖率测试',
            'schedule': '任务调度',
            'numpy': '数值计算',
            'pandas': '数据处理',
            'requests': 'HTTP请求',
            'asyncio': '异步编程',
            'pathlib': '路径处理',
            'logging': '日志记录'
        }

        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package)
                print(f"  ✅ {package}: {description}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}: {description} (缺失)")

        if missing_packages:
            print(f"\n❌ 缺少必要的Python包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

        print("✅ 所有Python包依赖检查通过")
        return True

    def check_system_directories(self) -> bool:
        """检查系统目录"""
        print("📁 检查系统目录...")

        required_dirs = [
            'src',
            'tests',
            'logs',
            'cache',
            'reports',
            'data'
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                print(f"  ❌ {dir_name}: 目录不存在")
            else:
                print(f"  ✅ {dir_name}: 目录存在")

        if missing_dirs:
            print(f"\n❌ 缺少必要目录: {', '.join(missing_dirs)}")
            print("请创建缺失的目录")
            return False

        print("✅ 所有系统目录检查通过")
        return True

    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        print("💾 检查磁盘空间...")

        try:
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)

            print(f"  总空间: {total_gb:.2f}GB")
            print(f"  已使用: {used / (1024**3):.2f}GB")
            print(f"  可用空间: {free_gb:.2f}GB")

            if free_gb < 1.0:
                print("❌ 磁盘空间不足 (少于1GB)")
                return False
            elif free_gb < 5.0:
                print("⚠️ 磁盘空间较少 (少于5GB)")
            else:
                print("✅ 磁盘空间充足")

            return True
        except Exception as e:
            print(f"⚠️ 无法检查磁盘空间: {e}")
            return True  # 不阻止运行

    def check_network_connectivity(self) -> bool:
        """检查网络连接"""
        print("🌐 检查网络连接...")

        test_urls = [
            'http://localhost:11434',  # 本地AI服务
            'https://pypi.org',        # PyPI
            'https://github.com'        # GitHub
        ]

        for url in test_urls:
            try:
                import requests
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ {url}: 连接正常")
                else:
                    print(f"  ⚠️ {url}: 响应异常 ({response.status_code})")
            except Exception as e:
                print(f"  ❌ {url}: 连接失败 ({e})")

        print("✅ 网络连接检查完成")
        return True

    async def check_ai_service(self, api_base: str = "http://localhost:11434") -> bool:
        """检查AI服务"""
        print("🤖 检查AI服务...")

        try:
            async with aiohttp.ClientSession() as session:
                # 检查服务可用性
                async with session.get(
                    f"{api_base}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            models = [model.get('id', 'unknown') for model in data['data']]
                            print(f"  ✅ AI服务正常，可用模型: {', '.join(models)}")
                            return True
                        else:
                            print("  ❌ AI服务响应格式异常")
                            return False
                    else:
                        print(f"  ❌ AI服务响应异常: {response.status}")
                        return False
        except asyncio.TimeoutError:
            print("  ❌ AI服务连接超时")
            return False
        except Exception as e:
            print(f"  ❌ AI服务连接失败: {e}")
            return False

    def check_pytest_installation(self) -> bool:
        """检查pytest安装"""
        print("🧪 检查pytest安装...")

        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  ✅ pytest: {version}")
                return True
            else:
                print(f"  ❌ pytest检查失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ❌ pytest检查异常: {e}")
            return False

    def check_coverage_tools(self) -> bool:
        """检查覆盖率工具"""
        print("📊 检查覆盖率工具...")

        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--cov', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print("  ✅ pytest-cov: 已安装")
                return True
            else:
                print("  ❌ pytest-cov: 未安装或配置错误")
                return False
        except Exception as e:
            print(f"  ❌ 覆盖率工具检查异常: {e}")
            return False

    def check_system_resources(self) -> bool:
        """检查系统资源"""
        print("💻 检查系统资源...")

        import psutil

        # 检查内存
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_percent = memory.percent

        print(f"  内存: {memory_gb:.2f}GB (使用率: {memory_percent:.1f}%)")

        if memory_percent > 90:
            print("  ⚠️ 内存使用率过高")
        elif memory_percent > 80:
            print("  ⚠️ 内存使用率较高")
        else:
            print("  ✅ 内存使用正常")

        # 检查CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"  CPU使用率: {cpu_percent:.1f}%")

        if cpu_percent > 90:
            print("  ⚠️ CPU使用率过高")
        elif cpu_percent > 80:
            print("  ⚠️ CPU使用率较高")
        else:
            print("  ✅ CPU使用正常")

        return True

    def generate_report(self) -> str:
        """生成检查报告"""
        report_file = "reports/testing/dependency_check_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report_content = f"""# RQA2025 AI覆盖率自动化依赖检查报告

## 📊 检查摘要

**检查时间**: {current_time}
**系统平台**: {platform.system()} {platform.release()}
**Python版本**: {sys.version}

## 🔍 检查结果

"""

        for check_name, result in self.check_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            report_content += f"- **{check_name}**: {status}\n"

        report_content += f"""
## 📋 详细结果

### 系统要求
- Python版本: {'✅ 满足要求' if self.check_results.get('python_version', False) else '❌ 版本过低'}
- conda环境: {'✅ 已激活' if self.check_results.get('conda_environment', False) else '❌ 未激活'}
- 磁盘空间: {'✅ 充足' if self.check_results.get('disk_space', False) else '❌ 不足'}

### Python依赖
- 包依赖: {'✅ 完整' if self.check_results.get('python_packages', False) else '❌ 缺失'}
- pytest: {'✅ 已安装' if self.check_results.get('pytest_installation', False) else '❌ 未安装'}
- 覆盖率工具: {'✅ 已安装' if self.check_results.get('coverage_tools', False) else '❌ 未安装'}

### 外部服务
- AI服务: {'✅ 可用' if self.check_results.get('ai_service', False) else '❌ 不可用'}
- 网络连接: {'✅ 正常' if self.check_results.get('network_connectivity', False) else '❌ 异常'}

## 🚀 建议

"""

        if all(self.check_results.values()):
            report_content += "✅ 所有检查通过，可以开始使用AI覆盖率自动化功能"
        else:
            failed_checks = [name for name, result in self.check_results.items() if not result]
            report_content += f"❌ 以下检查失败，请修复后重试:\n"
            for check in failed_checks:
                report_content += f"- {check}\n"

        report_content += f"""
## 📞 故障排除

1. **Python版本问题**: 确保使用Python 3.8或更高版本
2. **conda环境问题**: 运行 `conda activate rqa`
3. **包依赖问题**: 运行 `pip install aiohttp pytest pytest-cov schedule`
4. **AI服务问题**: 确保Deepseek服务正在运行
5. **磁盘空间问题**: 清理不必要的文件释放空间

---
**报告版本**: v1.0
**检查时间**: {current_time}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return report_file

    async def run_all_checks(self) -> bool:
        """运行所有检查"""
        print("🔍 开始依赖检查...")
        print("=" * 50)

        # 运行各项检查
        checks = [
            ("Python版本", self.check_python_version),
            ("conda环境", self.check_conda_environment),
            ("Python包", self.check_python_packages),
            ("系统目录", self.check_system_directories),
            ("磁盘空间", self.check_disk_space),
            ("网络连接", self.check_network_connectivity),
            ("pytest安装", self.check_pytest_installation),
            ("覆盖率工具", self.check_coverage_tools),
            ("系统资源", self.check_system_resources),
        ]

        for check_name, check_func in checks:
            print(f"\n📋 检查: {check_name}")
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                self.check_results[check_name] = result
            except Exception as e:
                print(f"❌ 检查异常: {e}")
                self.check_results[check_name] = False

        # 检查AI服务
        print(f"\n📋 检查: AI服务")
        try:
            ai_result = await self.check_ai_service()
            self.check_results['ai_service'] = ai_result
        except Exception as e:
            print(f"❌ AI服务检查异常: {e}")
            self.check_results['ai_service'] = False

        # 生成报告
        report_file = self.generate_report()

        # 显示结果
        print("\n" + "=" * 50)
        print("📊 检查结果汇总:")

        passed = sum(self.check_results.values())
        total = len(self.check_results)

        for check_name, result in self.check_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {check_name}: {status}")

        success_rate = (passed / total) * 100
        print(f"\n📈 总体结果: {passed}/{total} 通过 ({success_rate:.1f}%)")

        if success_rate >= 90:
            print("🎉 依赖检查通过，可以开始使用AI覆盖率自动化！")
            return True
        else:
            print("⚠️ 部分检查失败，请修复后重试")
            print(f"📄 详细报告: {report_file}")
            return False


async def main():
    """主函数"""
    print("🧪 RQA2025 AI覆盖率自动化依赖检查")
    print("=" * 50)

    checker = DependencyChecker()
    result = await checker.run_all_checks()

    if result:
        print("\n🎯 所有依赖检查通过，可以开始使用AI覆盖率自动化功能")
        print("\n📖 使用说明:")
        print("  python scripts/testing/start_ai_coverage_automation.py check")
        print("  python scripts/testing/start_ai_coverage_automation.py once")
        print("  python scripts/testing/start_ai_coverage_automation.py start --mode continuous")
    else:
        print("\n❌ 依赖检查失败，请修复问题后重试")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
