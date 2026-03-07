#!/usr/bin/env python3
"""
RQA2025 安全模块整合脚本

将分散在不同目录的安全模块整合到统一的src/security/目录中
"""

import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityModuleConsolidator:
    """安全模块整合器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup" / "security_modules"
        self.consolidated_dir = self.project_root / "src" / "security"
        self.consolidated_dir.mkdir(parents=True, exist_ok=True)

    def analyze_current_structure(self) -> Dict[str, Any]:
        """分析当前安全模块结构"""
        analysis = {
            "src_security": [],
            "data_security": [],
            "infrastructure_security": [],
            "duplicates": [],
            "recommendations": []
        }

        # 分析src/security目录
        if (self.project_root / "src" / "security").exists():
            analysis["src_security"] = [
                str(f.name) for f in (self.project_root / "src" / "security").glob("*.py")
                if not f.name.startswith("__")
            ]

        # 分析src/data/security目录
        if (self.project_root / "src" / "data" / "security").exists():
            analysis["data_security"] = [
                str(f.name) for f in (self.project_root / "src" / "data" / "security").glob("*.py")
                if not f.name.startswith("__")
            ]

        # 分析src/infrastructure/security目录
        if (self.project_root / "src" / "infrastructure" / "security").exists():
            analysis["infrastructure_security"] = [
                str(f.name) for f in (self.project_root / "src" / "infrastructure" / "security").glob("*.py")
                if not f.name.startswith("__")
            ]

        # 检测重复功能
        all_files = (analysis["src_security"] + analysis["data_security"] +
                     analysis["infrastructure_security"])

        file_names = all_files
        duplicates = set([name for name in file_names if file_names.count(name) > 1])
        analysis["duplicates"] = list(duplicates)

        # 生成建议
        if duplicates:
            analysis["recommendations"].append(
                f"发现重复文件: {', '.join(duplicates)}，建议保留最新的实现"
            )

        if len(analysis["src_security"]) == 0:
            analysis["recommendations"].append(
                "建议将所有安全模块整合到src/security/目录"
            )

        if len(analysis["data_security"]) > 0:
            analysis["recommendations"].append(
                "数据层安全模块建议迁移到统一的src/security/目录"
            )

        if len(analysis["infrastructure_security"]) > 0:
            analysis["recommendations"].append(
                "基础设施层安全模块建议迁移到统一的src/security/目录"
            )

        return analysis

    def create_backup(self) -> bool:
        """创建备份"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # 备份src/security目录
            if (self.project_root / "src" / "security").exists():
                shutil.copytree(
                    self.project_root / "src" / "security",
                    self.backup_dir / "src_security_backup",
                    dirs_exist_ok=True
                )

            # 备份src/data/security目录
            if (self.project_root / "src" / "data" / "security").exists():
                shutil.copytree(
                    self.project_root / "src" / "data" / "security",
                    self.backup_dir / "data_security_backup",
                    dirs_exist_ok=True
                )

            # 备份src/infrastructure/security目录
            if (self.project_root / "src" / "infrastructure" / "security").exists():
                shutil.copytree(
                    self.project_root / "src" / "infrastructure" / "security",
                    self.backup_dir / "infrastructure_security_backup",
                    dirs_exist_ok=True
                )

            logger.info(f"安全模块备份已创建到: {self.backup_dir}")
            return True

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False

    def create_unified_interfaces(self) -> bool:
        """创建统一的接口文件"""
        try:
            # 创建统一的__init__.py
            init_content = '''#!/usr/bin/env python3
"""
RQA2025 统一安全模块

提供统一的安全接口，包括：
- 审计系统 (audit_system.py)
- 访问控制系统 (access_control.py)
- 加密服务 (encryption_service.py)

所有安全功能都通过src/core/integration/security_adapter.py进行统一访问
"""

from .audit_system import get_audit_system, AuditSystem
from .access_control import get_access_control_system, AccessControlSystem
from .encryption_service import get_encryption_service, EncryptionService

__all__ = [
    'get_audit_system',
    'get_access_control_system',
    'get_encryption_service',
    'AuditSystem',
    'AccessControlSystem',
    'EncryptionService'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "RQA2025 Team"
'''

            with open(self.consolidated_dir / "__init__.py", 'w', encoding='utf-8') as f:
                f.write(init_content)

            # 创建README.md
            readme_content = '''# RQA2025 统一安全模块

## 概述

本目录包含RQA2025项目的统一安全模块实现，整合了所有安全相关的功能。

## 模块说明

### audit_system.py
- 审计系统：记录和监控所有安全相关的事件
- 功能：操作审计、安全事件记录、风险评估

### access_control.py
- 访问控制系统：基于角色的访问控制(RBAC)
- 功能：用户认证、权限管理、会话管理

### encryption_service.py
- 加密服务：数据加密和解密
- 功能：数据加密、数字签名、安全令牌

## 使用方式

### 通过统一基础设施集成层访问

```python
from src.core.integration.security_adapter import get_security_integration_manager

# 获取安全集成管理器
security_manager = get_security_integration_manager()

# 审计操作
security_manager.audit_operation("login", "user123", "auth_system", "success")

# 安全检查
clearance = security_manager.check_security_clearance("user123", "/api/admin", "read")

# 数据加密
encrypted = security_manager.encrypt_sensitive_data("sensitive_data")
```

### 直接访问各个模块

```python
from src.security import get_audit_system, get_access_control_system, get_encryption_service

# 获取各个系统实例
audit = get_audit_system()
access_control = get_access_control_system()
encryption = get_encryption_service()
```

## 架构原则

1. **统一管理**：所有安全功能通过统一接口访问
2. **模块化设计**：各功能模块独立，接口清晰
3. **高可用性**：支持降级服务和故障转移
4. **可扩展性**：易于添加新的安全功能

## 安全级别

- **LOW**: 基本安全功能
- **MEDIUM**: 标准安全功能
- **HIGH**: 高级安全功能
- **CRITICAL**: 关键安全功能

## 注意事项

1. 所有敏感操作都会被记录到审计日志
2. 加密密钥需要定期轮换
3. 定期检查和更新访问权限
4. 监控安全事件和异常情况
'''

            with open(self.consolidated_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)

            logger.info("统一接口文件已创建")
            return True

        except Exception as e:
            logger.error(f"创建统一接口文件失败: {e}")
            return False

    def generate_migration_report(self) -> Dict[str, Any]:
        """生成迁移报告"""
        analysis = self.analyze_current_structure()

        report = {
            "timestamp": str(datetime.now()),
            "analysis": analysis,
            "migration_plan": {
                "backup_created": False,
                "files_to_migrate": {
                    "data_security": analysis["data_security"],
                    "infrastructure_security": analysis["infrastructure_security"]
                },
                "consolidation_strategy": "keep_latest_implementation",
                "integration_approach": "unified_adapter_pattern"
            },
            "risks": [
                "功能重复可能导致冲突",
                "接口不一致性",
                "测试覆盖不足",
                "配置迁移复杂"
            ],
            "benefits": [
                "统一管理减少维护成本",
                "消除重复代码",
                "提高系统一致性",
                "便于功能扩展"
            ]
        }

        return report

    def save_migration_report(self, report_path: Optional[str] = None) -> bool:
        """保存迁移报告"""
        if not report_path:
            report_path = self.project_root / "security_migration_report.json"

        try:
            report = self.generate_migration_report()
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"迁移报告已保存到: {report_path}")
            return True

        except Exception as e:
            logger.error(f"保存迁移报告失败: {e}")
            return False


def main():
    """主函数"""
    print("🚀 RQA2025 安全模块整合分析工具")
    print("="*50)

    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 创建整合器
    consolidator = SecurityModuleConsolidator(project_root)

    # 分析当前结构
    print("\n📊 分析当前安全模块结构...")
    analysis = consolidator.analyze_current_structure()

    print("\n发现的安全模块:")
    print(f"  • src/security/: {len(analysis['src_security'])} 个文件")
    print(f"  • src/data/security/: {len(analysis['data_security'])} 个文件")
    print(f"  • src/infrastructure/security/: {len(analysis['infrastructure_security'])} 个文件")

    if analysis['duplicates']:
        print(f"\n⚠️  发现重复文件: {', '.join(analysis['duplicates'])}")

    print("\n📋 建议:")
    for recommendation in analysis['recommendations']:
        print(f"  • {recommendation}")

    # 创建备份
    print("\n💾 创建备份...")
    if consolidator.create_backup():
        print("✅ 备份创建成功")
    else:
        print("❌ 备份创建失败")

    # 创建统一接口
    print("\n🔧 创建统一接口...")
    if consolidator.generate_migration_report():
        print("✅ 统一接口创建成功")
    else:
        print("❌ 统一接口创建失败")

    # 保存迁移报告
    print("\n📄 生成迁移报告...")
    if consolidator.save_migration_report():
        print("✅ 迁移报告生成成功")
    else:
        print("❌ 迁移报告生成失败")

    print("\n" + "="*50)
    print("🎯 整合建议:")
    print("1. 保留 src/security/ 目录作为统一安全模块目录")
    print("2. 将 src/data/security/ 中的文件手动检查并整合")
    print("3. 将 src/infrastructure/security/ 中的文件手动检查并整合")
    print("4. 更新所有导入语句使用统一的接口")
    print("5. 运行完整的测试套件验证整合结果")
    print("\n📝 详细的迁移报告已保存到: security_migration_report.json")


if __name__ == "__main__":
    main()
