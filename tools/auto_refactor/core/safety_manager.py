#!/usr/bin/env python3
"""
安全管理器

负责重构过程的安全保障，包括备份、验证、回滚等功能。
"""

import os
import shutil
import hashlib
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .config import RefactorConfig, SafetyLevel


@dataclass
class BackupResult:
    """备份结果"""

    success: bool
    backup_path: Optional[str] = None
    original_hash: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """验证结果"""

    success: bool
    checks_performed: List[str] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.checks_performed is None:
            self.checks_performed = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class RollbackResult:
    """回滚结果"""

    success: bool
    original_restored: bool = False
    backup_removed: bool = False
    error: Optional[str] = None


class BackupManager:
    """备份管理器"""

    def __init__(self, config: RefactorConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir) if config.backup_dir else Path(
            tempfile.gettempdir()) / "auto_refactor_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, file_path: str) -> BackupResult:
        """
        创建文件备份

        Args:
            file_path: 要备份的文件路径

        Returns:
            备份结果
        """
        try:
            source_path = Path(file_path)

            if not source_path.exists():
                return BackupResult(
                    success=False,
                    error=f"File does not exist: {file_path}"
                )

            # 计算原文件哈希
            original_hash = self._calculate_file_hash(source_path)

            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = source_path.stem
            file_suffix = source_path.suffix
            backup_filename = f"{file_stem}_{timestamp}_backup{file_suffix}"
            backup_path = self.backup_dir / backup_filename

            # 复制文件
            shutil.copy2(source_path, backup_path)

            # 验证备份
            backup_hash = self._calculate_file_hash(backup_path)
            if original_hash != backup_hash:
                # 删除失败的备份
                backup_path.unlink(missing_ok=True)
                return BackupResult(
                    success=False,
                    error="Backup verification failed: hash mismatch"
                )

            return BackupResult(
                success=True,
                backup_path=str(backup_path),
                original_hash=original_hash
            )

        except Exception as e:
            return BackupResult(
                success=False,
                error=f"Backup creation failed: {str(e)}"
            )

    def rollback_backup(self, file_path: str, backup_path: Optional[str] = None) -> RollbackResult:
        """
        回滚到备份版本

        Args:
            file_path: 要回滚的文件路径
            backup_path: 备份文件路径，如果不提供则自动查找最新的备份

        Returns:
            回滚结果
        """
        try:
            target_path = Path(file_path)

            if backup_path is None:
                # 查找最新的备份文件
                backup_path = self._find_latest_backup(file_path)
                if not backup_path:
                    return RollbackResult(
                        success=False,
                        error="No backup found for file"
                    )

            backup_path = Path(backup_path)

            if not backup_path.exists():
                return RollbackResult(
                    success=False,
                    error=f"Backup file does not exist: {backup_path}"
                )

            # 备份当前文件（以防回滚失败）
            emergency_backup = self._create_emergency_backup(target_path)

            try:
                # 执行回滚
                shutil.copy2(backup_path, target_path)

                # 验证回滚结果
                if self._verify_file_integrity(target_path, backup_path):
                    # 清理紧急备份
                    if emergency_backup:
                        emergency_backup.unlink(missing_ok=True)

                    return RollbackResult(
                        success=True,
                        original_restored=True,
                        backup_removed=False  # 保留备份用于审计
                    )
                else:
                    # 恢复紧急备份
                    if emergency_backup:
                        shutil.copy2(emergency_backup, target_path)
                        emergency_backup.unlink(missing_ok=True)

                    return RollbackResult(
                        success=False,
                        error="Rollback verification failed"
                    )

            except Exception as rollback_error:
                # 恢复紧急备份
                if emergency_backup and emergency_backup.exists():
                    try:
                        shutil.copy2(emergency_backup, target_path)
                    except Exception:
                        pass  # 忽略恢复失败的错误
                    emergency_backup.unlink(missing_ok=True)

                return RollbackResult(
                    success=False,
                    error=f"Rollback failed: {str(rollback_error)}"
                )

        except Exception as e:
            return RollbackResult(
                success=False,
                error=f"Rollback operation failed: {str(e)}"
            )

    def cleanup_old_backups(self, max_age_days: int = 7, max_backups_per_file: int = 5):
        """
        清理旧备份文件

        Args:
            max_age_days: 最大保留天数
            max_backups_per_file: 每个文件最多保留的备份数
        """
        try:
            now = datetime.now()
            file_backups = {}  # file_stem -> list of backup files

            # 收集所有备份文件
            for backup_file in self.backup_dir.glob("*_backup*"):
                if not backup_file.is_file():
                    continue

                try:
                    # 解析文件名获取原文件名
                    name_parts = backup_file.stem.split('_')
                    if len(name_parts) < 3 or name_parts[-1] != 'backup':
                        continue

                    original_stem = '_'.join(name_parts[:-2])  # 移除时间戳和backup
                    file_backups.setdefault(original_stem, []).append(backup_file)

                except Exception:
                    continue  # 跳过无法解析的文件名

            # 清理每个文件的备份
            for original_stem, backups in file_backups.items():
                # 按修改时间排序（最新的在前）
                backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # 删除超出数量限制的备份
                for old_backup in backups[max_backups_per_file:]:
                    try:
                        old_backup.unlink()
                    except Exception:
                        pass  # 忽略删除失败

                # 删除过期的备份
                for backup in backups:
                    try:
                        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                        if (now - mtime).days > max_age_days:
                            backup.unlink()
                    except Exception:
                        pass  # 忽略删除失败

        except Exception as e:
            print(f"Warning: Backup cleanup failed: {e}")

    def _find_latest_backup(self, file_path: str) -> Optional[str]:
        """查找文件的最新备份"""
        try:
            source_path = Path(file_path)
            file_stem = source_path.stem

            # 查找匹配的备份文件
            matching_backups = []
            for backup_file in self.backup_dir.glob(f"{file_stem}_*_backup{source_path.suffix}"):
                try:
                    # 验证文件名格式
                    name_parts = backup_file.stem.split('_')
                    if len(name_parts) >= 3 and name_parts[-1] == 'backup':
                        matching_backups.append(backup_file)
                except Exception:
                    continue

            if not matching_backups:
                return None

            # 返回最新的备份
            return str(max(matching_backups, key=lambda x: x.stat().st_mtime))

        except Exception:
            return None

    def _create_emergency_backup(self, file_path: Path) -> Optional[Path]:
        """创建紧急备份（用于回滚失败时的恢复）"""
        try:
            if not file_path.exists():
                return None

            emergency_name = f"{file_path.stem}_emergency_backup{file_path.suffix}"
            emergency_path = self.backup_dir / emergency_name

            shutil.copy2(file_path, emergency_path)
            return emergency_path

        except Exception:
            return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _verify_file_integrity(self, file1: Path, file2: Path) -> bool:
        """验证两个文件是否相同"""
        try:
            return self._calculate_file_hash(file1) == self._calculate_file_hash(file2)
        except Exception:
            return False


class ValidationManager:
    """验证管理器"""

    def __init__(self, config: RefactorConfig):
        self.config = config

    def validate_syntax(self, file_path: str) -> ValidationResult:
        """
        验证文件语法

        Args:
            file_path: 文件路径

        Returns:
            验证结果
        """
        result = ValidationResult(success=True)
        result.checks_performed.append("syntax_check")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # 使用AST验证语法
            import ast
            ast.parse(source_code, filename=file_path)

        except SyntaxError as e:
            result.success = False
            result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result.success = False
            result.errors.append(f"Syntax validation failed: {str(e)}")

        return result

    def validate_imports(self, file_path: str) -> ValidationResult:
        """
        验证导入语句

        Args:
            file_path: 文件路径

        Returns:
            验证结果
        """
        result = ValidationResult(success=True)
        result.checks_performed.append("import_check")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # 解析AST并检查导入
            import ast
            tree = ast.parse(source_code, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # 这里可以添加更复杂的导入验证逻辑
                    # 例如检查模块是否存在、循环导入等
                    pass

        except Exception as e:
            result.success = False
            result.errors.append(f"Import validation failed: {str(e)}")

        return result

    def validate_semantics(self, file_path: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        验证语义正确性

        Args:
            file_path: 文件路径
            context: 验证上下文

        Returns:
            验证结果
        """
        result = ValidationResult(success=True)
        result.checks_performed.append("semantic_check")

        try:
            # 这里可以实现更复杂的语义验证
            # 例如类型检查、变量使用检查等
            # 暂时只做基本的AST验证

            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            import ast
            tree = ast.parse(source_code, filename=file_path)

            # 检查基本语义问题
            semantic_issues = self._check_semantic_issues(tree)
            result.warnings.extend(semantic_issues)

        except Exception as e:
            result.success = False
            result.errors.append(f"Semantic validation failed: {str(e)}")

        return result

    def _check_semantic_issues(self, tree: ast.AST) -> List[str]:
        """检查语义问题"""
        issues = []

        # 检查未使用的变量（简化实现）
        defined_vars = set()
        used_vars = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)

        # 查找可能的未使用变量（这只是一个简单的启发式检查）
        potentially_unused = defined_vars - used_vars
        for var in potentially_unused:
            if not var.startswith('_'):  # 忽略以下划线开头的变量
                issues.append(f"Potentially unused variable: {var}")

        return issues

    def run_all_validations(self, file_path: str) -> ValidationResult:
        """
        运行所有验证

        Args:
            file_path: 文件路径

        Returns:
            综合验证结果
        """
        all_checks = []
        all_errors = []
        all_warnings = []

        # 语法验证
        syntax_result = self.validate_syntax(file_path)
        all_checks.extend(syntax_result.checks_performed or [])
        all_errors.extend(syntax_result.errors or [])
        all_warnings.extend(syntax_result.warnings or [])

        # 如果语法验证失败，后续验证就没有意义
        if not syntax_result.success:
            return ValidationResult(
                success=False,
                checks_performed=all_checks,
                errors=all_errors,
                warnings=all_warnings
            )

        # 导入验证
        if self.config.import_validation:
            import_result = self.validate_imports(file_path)
            all_checks.extend(import_result.checks_performed or [])
            all_errors.extend(import_result.errors or [])
            all_warnings.extend(import_result.warnings or [])

        # 语义验证
        if self.config.semantic_validation:
            semantic_result = self.validate_semantics(file_path)
            all_checks.extend(semantic_result.checks_performed or [])
            all_errors.extend(semantic_result.errors or [])
            all_warnings.extend(semantic_result.warnings or [])

        return ValidationResult(
            success=len(all_errors) == 0,
            checks_performed=all_checks,
            errors=all_errors,
            warnings=all_warnings
        )


class SafetyManager:
    """安全管理器"""

    def __init__(self, config: RefactorConfig):
        self.config = config
        self.backup_manager = BackupManager(config)
        self.validation_manager = ValidationManager(config)

    def check_safety(self, suggestion) -> bool:
        """
        检查重构建议的安全性

        Args:
            suggestion: 重构建议

        Returns:
            是否安全
        """
        # 根据安全级别进行不同的检查
        if self.config.safety_level == SafetyLevel.LOW:
            return True  # 低安全级别不做额外检查

        # 检查文件是否存在
        if not os.path.exists(suggestion.file_path):
            return False

        # 检查文件是否可写
        if not os.access(suggestion.file_path, os.W_OK):
            return False

        # 检查建议的置信度
        if self.config.safety_level == SafetyLevel.HIGH:
            return suggestion.confidence >= 0.8
        elif self.config.safety_level == SafetyLevel.MEDIUM:
            return suggestion.confidence >= 0.6

        return True

    def create_backup(self, file_path: str) -> BackupResult:
        """创建备份"""
        return self.backup_manager.create_backup(file_path)

    def validate_refactor(self, file_path: str, refactor_result) -> ValidationResult:
        """验证重构结果"""
        return self.validation_manager.run_all_validations(file_path)

    def rollback_backup(self, file_path: str) -> RollbackResult:
        """回滚备份"""
        return self.backup_manager.rollback_backup(file_path)

    def cleanup_backups(self):
        """清理旧备份"""
        self.backup_manager.cleanup_old_backups()
