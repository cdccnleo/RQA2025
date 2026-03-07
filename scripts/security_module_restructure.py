#!/usr/bin/env python3
"""
RQA2025 安全模块重构脚本

将基础设施层安全模块作为统一安全模块的核心，进行重组整合
"""

import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityModuleRestructure:
    """安全模块重构器"""

    def __init__(self, project_root: str, target_path: str = "core"):
        self.project_root = Path(project_root)
        self.source_dir = self.project_root / "src" / "infrastructure" / "security"

        # 处理目标路径
        if target_path == "security":
            # 顶级目录
            self.target_dir = self.project_root / "src" / "security"
        elif target_path == "infrastructure/security":
            # 保持原路径
            self.target_dir = self.project_root / "src" / "infrastructure" / "security"
        else:
            # core层
            self.target_dir = self.project_root / "src" / "core" / "security"

        self.backup_dir = self.project_root / "backup" / "security_restructure"
        self.data_security_dir = self.project_root / "src" / "data" / "security"
        self.target_path = target_path

        # 确保目标目录存在
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> bool:
        """创建备份"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # 备份现有的安全模块
            backup_dirs = [
                ("infrastructure_security", self.source_dir),
                ("data_security", self.data_security_dir)
            ]

            # 只在目标目录不是当前目录时才备份当前目录
            if self.target_dir != self.project_root / "src" / "security":
                backup_dirs.append(("current_security", self.target_dir))

            for backup_name, source_path in backup_dirs:
                if source_path.exists():
                    target_path = self.backup_dir / backup_name
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    logger.info(f"已备份 {source_path} 到 {target_path}")

            logger.info(f"安全模块备份完成: {self.backup_dir}")
            return True

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False

    def restructure_directory(self) -> bool:
        """重构目录结构"""
        try:
            logger.info("开始重构目录结构...")

            # 1. 清理目标目录（保留我们实现的模块）
            keep_files = ["audit_system.py", "access_control.py", "encryption_service.py"]
            for item in self.target_dir.iterdir():
                if item.is_file() and item.name not in keep_files and not item.name.startswith("__"):
                    item.unlink()
                    logger.info(f"删除旧文件: {item.name}")

            # 2. 创建子目录结构
            components_dir = self.target_dir / "components"
            services_dir = self.target_dir / "services"

            components_dir.mkdir(exist_ok=True)
            services_dir.mkdir(exist_ok=True)

            # 3. 从基础设施层迁移核心文件
            core_files = [
                "unified_security.py",
                "authentication_service.py",
                "base_security.py",
                "security_factory.py",
                "security_utils.py"
            ]

            for file_name in core_files:
                src_file = self.source_dir / file_name
                dst_file = self.target_dir / file_name

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"迁移核心文件: {file_name}")

            # 4. 迁移组件文件
            component_files = [
                "audit_components.py",
                "auth_components.py",
                "encrypt_components.py",
                "policy_components.py",
                "security_components.py"
            ]

            for file_name in component_files:
                src_file = self.source_dir / file_name
                dst_file = components_dir / file_name

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"迁移组件文件: {file_name}")

            # 5. 迁移服务文件
            service_files = [
                "data_protection_service.py",
                "web_management_service.py",
                "config_encryption_service.py"
            ]

            for file_name in service_files:
                src_file = self.source_dir / file_name
                dst_file = services_dir / file_name

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"迁移服务文件: {file_name}")

            # 6. 整合数据层安全功能
            data_security_mapping = {
                "access_control_manager.py": "data_access_control.py",
                "audit_logging_manager.py": "data_audit_manager.py",
                "data_encryption_manager.py": "data_encryption_service.py"
            }

            for src_name, dst_name in data_security_mapping.items():
                src_file = self.data_security_dir / src_name
                dst_file = services_dir / dst_name

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"整合数据层安全文件: {src_name} -> {dst_name}")

            logger.info("目录结构重构完成")
            return True

        except Exception as e:
            logger.error(f"重构目录结构失败: {e}")
            return False

    def update_imports(self) -> bool:
        """更新导入语句"""
        try:
            logger.info("开始更新导入语句...")

            # 需要更新的文件映射
            update_files = [
                # 核心文件
                self.target_dir / "unified_security.py",
                self.target_dir / "authentication_service.py",
                self.target_dir / "base_security.py",

                # 组件文件
                self.target_dir / "components" / "audit_components.py",
                self.target_dir / "components" / "auth_components.py",
                self.target_dir / "components" / "encrypt_components.py",

                # 服务文件
                self.target_dir / "services" / "data_protection_service.py",
                self.target_dir / "services" / "web_management_service.py",
            ]

            # 导入路径映射
            import_mappings = {
                "from .base_security import": "from src.core.security.base_security import",
                "from .interfaces import": "from src.core.security.interfaces import",
                "from .security_utils import": "from src.core.security.security_utils import",
                "from .security_components import": "from src.core.security.security_components import",
                "from .audit_components import": "from src.core.security.audit_components import",
                "from .auth_components import": "from src.core.security.auth_components import",
                "from .encrypt_components import": "from src.core.security.encrypt_components import",
            }

            for file_path in update_files:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 更新导入语句
                    for old_import, new_import in import_mappings.items():
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                            logger.info(f"更新导入: {file_path.name} - {old_import}")

                    # 保存更新后的文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)

            logger.info("导入语句更新完成")
            return True

        except Exception as e:
            logger.error(f"更新导入语句失败: {e}")
            return False

    def create_unified_init(self) -> bool:
        """创建统一的__init__.py文件"""
        try:
            init_content = '''#!/usr/bin/env python3
"""
RQA2025 统一安全模块

基于基础设施层安全模块重构，提供完整的统一安全功能
"""

from .unified_security import UnifiedSecurity, get_security, set_security
from .authentication_service import AuthenticationService, IAuthenticator
from .base_security import ISecurityComponent, SecurityLevel
from .audit_system import get_audit_system, AuditSystem
from .access_control import get_access_control_system, AccessControlSystem
from .encryption_service import get_encryption_service, EncryptionService

# 便捷导入
__all__ = [
    # 核心安全类
    'UnifiedSecurity',
    'AuthenticationService',
    'AuditSystem',
    'AccessControlSystem',
    'EncryptionService',

    # 接口
    'ISecurityComponent',
    'IAuthenticator',

    # 枚举
    'SecurityLevel',

    # 便捷函数
    'get_security',
    'set_security',
    'get_audit_system',
    'get_access_control_system',
    'get_encryption_service'
]

# 版本信息
__version__ = "2.0.0"
__author__ = "RQA2025 Team"

# 模块描述
__doc__ = """
统一安全模块提供：
- 用户认证和授权
- 数据加密和解密
- 操作审计和监控
- 访问控制和权限管理
- 安全策略和规则
"""

def initialize_security():
    """初始化安全模块"""
    # 这里可以添加安全模块的初始化逻辑
    pass

def health_check():
    """安全模块健康检查"""
    try:
        security = get_security()
        audit = get_audit_system()
        access_control = get_access_control_system()
        encryption = get_encryption_service()

        return {
            "status": "healthy",
            "modules": {
                "unified_security": security is not None,
                "audit_system": audit is not None,
                "access_control": access_control is not None,
                "encryption_service": encryption is not None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
'''

            init_file = self.target_dir / "__init__.py"
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(init_content)

            logger.info("统一__init__.py文件创建完成")
            return True

        except Exception as e:
            logger.error(f"创建统一__init__.py失败: {e}")
            return False

    def update_integration_adapter(self) -> bool:
        """更新集成适配器"""
        try:
            adapter_file = self.project_root / "src" / "core" / "integration" / "security_adapter.py"

            if adapter_file.exists():
                with open(adapter_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 添加对新重构模块的支持
                integration_updates = '''
    def get_unified_security(self):
        """获取统一安全服务"""
        try:
            from src.security.unified_security import get_security
            logger.info("使用统一安全服务")
            return get_security()
        except ImportError:
            logger.warning("统一安全服务导入失败")
            return None

    def get_authentication_service(self):
        """获取认证服务"""
        try:
            from src.security.authentication_service import AuthenticationService
            logger.info("使用认证服务")
            return AuthenticationService()
        except ImportError:
            logger.warning("认证服务导入失败")
            return None
'''

                # 在文件末尾添加新的方法
                if integration_updates not in content:
                    content += "\n" + integration_updates

                    with open(adapter_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    logger.info("集成适配器更新完成")

            return True

        except Exception as e:
            logger.error(f"更新集成适配器失败: {e}")
            return False

    def run_tests(self) -> bool:
        """运行测试验证重构结果"""
        try:
            logger.info("运行重构验证测试...")

            # 运行安全基础设施测试
            test_cmd = [
                "python",
                str(self.project_root / "tests" / "integration" /
                    "security" / "test_security_infrastructure.py")
            ]

            result = subprocess.run(test_cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("重构验证测试通过")
                return True
            else:
                logger.error(f"重构验证测试失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"运行测试失败: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """生成重构报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "restructure_type": "infrastructure_security_as_core",
            "phases": {
                "backup": False,
                "directory_restructure": False,
                "import_updates": False,
                "unified_init": False,
                "integration_update": False,
                "tests": False
            },
            "files_migrated": {
                "core_files": [],
                "component_files": [],
                "service_files": [],
                "data_security_files": []
            },
            "issues": [],
            "recommendations": []
        }

        return report

    def execute_restructure(self) -> bool:
        """执行完整重构流程"""
        logger.info("🚀 开始安全模块重构...")

        report = self.generate_report()

        # Phase 1: 备份
        logger.info("📦 Phase 1: 创建备份...")
        if self.create_backup():
            report["phases"]["backup"] = True
        else:
            report["issues"].append("备份创建失败")
            return False

        # Phase 2: 目录重构
        logger.info("🔧 Phase 2: 重构目录结构...")
        if self.restructure_directory():
            report["phases"]["directory_restructure"] = True

            # 更新文件列表
            if (self.target_dir / "unified_security.py").exists():
                report["files_migrated"]["core_files"].append("unified_security.py")
            if (self.target_dir / "components" / "audit_components.py").exists():
                report["files_migrated"]["component_files"].append("audit_components.py")
            if (self.target_dir / "services" / "data_protection_service.py").exists():
                report["files_migrated"]["service_files"].append("data_protection_service.py")
        else:
            report["issues"].append("目录重构失败")
            return False

        # Phase 3: 更新导入
        logger.info("📝 Phase 3: 更新导入语句...")
        if self.update_imports():
            report["phases"]["import_updates"] = True
        else:
            report["issues"].append("导入更新失败")

        # Phase 4: 创建统一接口
        logger.info("🔗 Phase 4: 创建统一接口...")
        if self.create_unified_init():
            report["phases"]["unified_init"] = True
        else:
            report["issues"].append("统一接口创建失败")

        # Phase 5: 更新集成适配器
        logger.info("🔌 Phase 5: 更新集成适配器...")
        if self.update_integration_adapter():
            report["phases"]["integration_update"] = True
        else:
            report["issues"].append("集成适配器更新失败")

        # Phase 6: 运行测试
        logger.info("🧪 Phase 6: 运行验证测试...")
        if self.run_tests():
            report["phases"]["tests"] = True
        else:
            report["issues"].append("验证测试失败")

        # 保存报告
        report_file = self.project_root / "security_restructure_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 输出结果
        successful_phases = sum(1 for phase in report["phases"].values() if phase)
        total_phases = len(report["phases"])

        if successful_phases == total_phases:
            logger.info("🎉 安全模块重构完全成功！")
            logger.info(f"📊 重构报告已保存到: {report_file}")
            return True
        else:
            logger.warning(f"⚠️  重构部分成功: {successful_phases}/{total_phases} 个阶段完成")
            if report["issues"]:
                logger.warning("发现问题:")
                for issue in report["issues"]:
                    logger.warning(f"  • {issue}")
            return False


def main():
    """主函数"""
    print("🔄 RQA2025 安全模块重构工具")
    print("="*50)
    print("将基础设施层安全模块作为统一安全模块的核心进行重构")
    print()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 选择目标路径
    print("请选择统一安全模块的目标路径:")
    print("1. src/core/security/     ⭐ 推荐 - 作为核心功能")
    print("2. src/security/          - 保持当前顶级目录")
    print("3. src/infrastructure/security/ - 保持基础设施层")
    print()

    choice = input("请输入选择 (1/2/3) [默认:1]: ").strip()

    if choice == "2":
        target_path = "security"
        print("📁 选择: src/security/")
    elif choice == "3":
        target_path = "infrastructure/security"
        print("📁 选择: src/infrastructure/security/")
    else:
        target_path = "core/security"
        print("📁 选择: src/core/security/ ⭐")

    print()

    # 创建重构器
    restructurer = SecurityModuleRestructure(project_root, target_path)

    # 执行重构
    if restructurer.execute_restructure():
        print()
        print("✅ 重构完成！")
        print("📋 接下来请：")
        print("  1. 检查重构报告: security_restructure_report.json")
        print("  2. 运行完整测试套件验证功能")
        print("  3. 更新项目文档")
        print("  4. 通知团队成员")
        print()
        print(f"🎯 统一安全模块路径: src/{target_path}/")
    else:
        print()
        print("❌ 重构失败！")
        print("📋 请：")
        print("  1. 查看错误日志")
        print("  2. 从备份恢复: backup/security_restructure/")
        print("  3. 联系技术支持")


if __name__ == "__main__":
    main()
