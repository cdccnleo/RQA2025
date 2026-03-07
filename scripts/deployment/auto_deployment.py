#!/usr/bin/env python3
"""
RQA2025 自动化部署脚本
实现一键部署和回滚功能
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class AutoDeployer:
    """自动化部署器"""

    def __init__(self, environment: str = "production", auto_rollback: bool = True):
        self.environment = environment
        self.auto_rollback = auto_rollback
        self.project_root = project_root
        self.deployment_log = []
        self.start_time = datetime.now()

    def deploy(self) -> bool:
        """执行完整部署流程"""
        print(f"🚀 开始 {self.environment} 环境部署...")

        try:
            # 1. 环境检查
            if not self._check_environment():
                print("❌ 环境检查失败")
                return False

            # 2. 备份当前版本
            if not self._backup_current_version():
                print("❌ 备份失败")
                return False

            # 3. 构建新版本
            if not self._build_new_version():
                print("❌ 构建失败")
                return False

            # 4. 部署新版本
            if not self._deploy_new_version():
                print("❌ 部署失败")
                if self.auto_rollback:
                    print("🔄 开始自动回滚...")
                    self._rollback()
                return False

            # 5. 健康检查
            if not self._health_check():
                print("❌ 健康检查失败")
                if self.auto_rollback:
                    print("🔄 开始自动回滚...")
                    self._rollback()
                return False

            # 6. 性能测试
            if not self._performance_test():
                print("⚠️ 性能测试未通过，但部署继续")

            print("✅ 部署成功完成")
            return True

        except Exception as e:
            print(f"❌ 部署过程中发生错误: {e}")
            if self.auto_rollback:
                print("🔄 开始自动回滚...")
                self._rollback()
            return False

    def _check_environment(self) -> bool:
        """检查部署环境"""
        print("📋 检查部署环境...")

        try:
            # 运行环境检查脚本
            result = subprocess.run([
                'python', 'scripts/environment_checker.py',
                '--env', self.environment
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"环境检查失败: {result.stderr}")
                return False

            print("✅ 环境检查通过")
            return True

        except Exception as e:
            print(f"环境检查异常: {e}")
            return False

    def _backup_current_version(self) -> bool:
        """备份当前版本"""
        print("💾 备份当前版本...")

        try:
            backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.project_root / 'backup' / backup_dir

            # 创建备份目录
            backup_path.mkdir(parents=True, exist_ok=True)

            # 备份配置文件
            config_files = [
                'config/default.json',
                'config/production.json',
                'deploy/docker-compose.yml'
            ]

            for config_file in config_files:
                src_path = self.project_root / config_file
                if src_path.exists():
                    dst_path = backup_path / config_file
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src_path, dst_path)

            # 备份当前Docker镜像
            try:
                subprocess.run([
                    'docker', 'tag', 'rqa2025:latest', 'rqa2025:backup'
                ], check=True)
            except:
                pass  # 如果镜像不存在，跳过

            self.backup_path = backup_path
            print(f"✅ 备份完成: {backup_path}")
            return True

        except Exception as e:
            print(f"备份失败: {e}")
            return False

    def _build_new_version(self) -> bool:
        """构建新版本"""
        print("🔨 构建新版本...")

        try:
            # 构建Docker镜像
            result = subprocess.run([
                'docker', 'build',
                '-t', 'rqa2025:latest',
                '--build-arg', f'ENV={self.environment}',
                '.'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"构建失败: {result.stderr}")
                return False

            print("✅ 构建完成")
            return True

        except Exception as e:
            print(f"构建异常: {e}")
            return False

    def _deploy_new_version(self) -> bool:
        """部署新版本"""
        print("🚀 部署新版本...")

        try:
            # 停止现有服务
            try:
                subprocess.run([
                    'docker-compose', '-f', 'deploy/docker-compose.yml', 'down'
                ], check=True)
            except:
                pass  # 如果服务未运行，忽略错误

            # 启动新服务
            result = subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml', 'up', '-d'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"部署失败: {result.stderr}")
                return False

            # 等待服务启动
            print("⏳ 等待服务启动...")
            time.sleep(30)

            print("✅ 部署完成")
            return True

        except Exception as e:
            print(f"部署异常: {e}")
            return False

    def _health_check(self) -> bool:
        """健康检查"""
        print("🏥 执行健康检查...")

        try:
            # 检查服务状态
            result = subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml', 'ps'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"服务状态检查失败: {result.stderr}")
                return False

            # 检查API健康状态
            import requests
            try:
                response = requests.get('http://localhost:8000/health', timeout=10)
                if response.status_code == 200:
                    print("✅ API健康检查通过")
                else:
                    print(f"❌ API健康检查失败: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ API健康检查异常: {e}")
                return False

            # 检查推理服务
            try:
                response = requests.get('http://localhost:8001/health', timeout=10)
                if response.status_code == 200:
                    print("✅ 推理服务健康检查通过")
                else:
                    print(f"❌ 推理服务健康检查失败: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ 推理服务健康检查异常: {e}")
                return False

            print("✅ 健康检查通过")
            return True

        except Exception as e:
            print(f"健康检查异常: {e}")
            return False

    def _performance_test(self) -> bool:
        """性能测试"""
        print("⚡ 执行性能测试...")

        try:
            # 运行性能测试脚本
            result = subprocess.run([
                'python', 'scripts/performance_test.py',
                '--duration', '5m',
                '--load', '100'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"性能测试失败: {result.stderr}")
                return False

            print("✅ 性能测试通过")
            return True

        except Exception as e:
            print(f"性能测试异常: {e}")
            return False

    def _rollback(self) -> bool:
        """回滚到上一个版本"""
        print("🔄 执行回滚...")

        try:
            # 停止当前服务
            try:
                subprocess.run([
                    'docker-compose', '-f', 'deploy/docker-compose.yml', 'down'
                ], check=True)
            except:
                pass

            # 恢复备份的镜像
            try:
                subprocess.run([
                    'docker', 'tag', 'rqa2025:backup', 'rqa2025:latest'
                ], check=True)
            except:
                pass

            # 启动备份的服务
            result = subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml', 'up', '-d'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"回滚失败: {result.stderr}")
                return False

            print("✅ 回滚完成")
            return True

        except Exception as e:
            print(f"回滚异常: {e}")
            return False

    def generate_deployment_report(self) -> Dict[str, Any]:
        """生成部署报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        report = {
            'deployment_id': f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'environment': self.environment,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'success': True,  # 如果到达这里，说明部署成功
            'logs': self.deployment_log
        }

        return report

    def save_deployment_report(self, report: Dict[str, Any], output_file: str = ""):
        """保存部署报告"""
        if not output_file:
            output_file = f"reports/deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 确保报告目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 部署报告已保存: {output_file}")
        return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 自动化部署脚本')
    parser.add_argument('--env', type=str, default='production', help='部署环境')
    parser.add_argument('--no-rollback', action='store_true', help='禁用自动回滚')
    parser.add_argument('--output', type=str, help='部署报告输出路径')

    args = parser.parse_args()

    # 创建部署器
    deployer = AutoDeployer(
        environment=args.env,
        auto_rollback=not args.no_rollback
    )

    # 执行部署
    success = deployer.deploy()

    # 生成部署报告
    report = deployer.generate_deployment_report()
    report['success'] = success

    # 保存报告
    output_file = args.output or f"reports/deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    deployer.save_deployment_report(report, output_file)

    # 显示部署摘要
    print(f"\n📈 部署摘要:")
    print(f"  环境: {report['environment']}")
    print(f"  部署ID: {report['deployment_id']}")
    print(f"  持续时间: {report['duration_seconds']:.1f}秒")
    print(f"  状态: {'✅ 成功' if success else '❌ 失败'}")

    # 返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
