#!/usr/bin/env python3
"""
自动化备份管理脚本
RQA2025 生产环境备份管理工具
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """备份配置类"""
    namespace: str = "rqa2025-production"
    backup_dir: str = "backups"
    retention_days: int = 30
    max_backups: int = 100
    compression: bool = True
    encryption: bool = True
    auto_cleanup: bool = True

class BackupManager:
    """备份管理器"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self) -> bool:
        """创建完整备份"""
        logger.info("💾 开始创建备份...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        try:
            # 1. 备份Kubernetes资源
            if not self._backup_k8s_resources(backup_path):
                logger.error("❌ Kubernetes资源备份失败")
                return False
            
            # 2. 备份配置映射
            if not self._backup_configmaps(backup_path):
                logger.error("❌ 配置映射备份失败")
                return False
            
            # 3. 备份密钥
            if not self._backup_secrets(backup_path):
                logger.error("❌ 密钥备份失败")
                return False
            
            # 4. 备份数据卷
            if not self._backup_persistent_volumes(backup_path):
                logger.error("❌ 数据卷备份失败")
                return False
            
            # 5. 备份应用数据
            if not self._backup_application_data(backup_path):
                logger.error("❌ 应用数据备份失败")
                return False
            
            # 6. 创建备份元数据
            self._create_backup_metadata(backup_path, timestamp)
            
            # 7. 压缩备份
            if self.config.compression:
                self._compress_backup(backup_path)
            
            # 8. 加密备份
            if self.config.encryption:
                self._encrypt_backup(backup_path)
            
            logger.info(f"✅ 备份创建成功: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 备份创建失败: {e}")
            return False
    
    def _backup_k8s_resources(self, backup_path: Path) -> bool:
        """备份Kubernetes资源"""
        logger.info("📋 备份Kubernetes资源...")
        
        try:
            # 备份所有资源
            result = subprocess.run([
                "kubectl", "get", "all", "-n", self.config.namespace,
                "-o", "yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                k8s_file = backup_path / "kubernetes_resources.yaml"
                with open(k8s_file, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                logger.info("✅ Kubernetes资源备份完成")
                return True
            else:
                logger.error(f"❌ Kubernetes资源备份失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Kubernetes资源备份异常: {e}")
            return False
    
    def _backup_configmaps(self, backup_path: Path) -> bool:
        """备份配置映射"""
        logger.info("⚙️ 备份配置映射...")
        
        try:
            result = subprocess.run([
                "kubectl", "get", "configmaps", "-n", self.config.namespace,
                "-o", "yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                config_file = backup_path / "configmaps.yaml"
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                logger.info("✅ 配置映射备份完成")
                return True
            else:
                logger.error(f"❌ 配置映射备份失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 配置映射备份异常: {e}")
            return False
    
    def _backup_secrets(self, backup_path: Path) -> bool:
        """备份密钥"""
        logger.info("🔐 备份密钥...")
        
        try:
            result = subprocess.run([
                "kubectl", "get", "secrets", "-n", self.config.namespace,
                "-o", "yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                secrets_file = backup_path / "secrets.yaml"
                with open(secrets_file, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                logger.info("✅ 密钥备份完成")
                return True
            else:
                logger.error(f"❌ 密钥备份失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 密钥备份异常: {e}")
            return False
    
    def _backup_persistent_volumes(self, backup_path: Path) -> bool:
        """备份持久化卷"""
        logger.info("💾 备份持久化卷...")
        
        try:
            # 获取PVC列表
            result = subprocess.run([
                "kubectl", "get", "pvc", "-n", self.config.namespace,
                "-o", "jsonpath={.items[*].metadata.name}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                pvc_names = result.stdout.strip().split()
                pvc_dir = backup_path / "persistent_volumes"
                pvc_dir.mkdir(exist_ok=True)
                
                for pvc_name in pvc_names:
                    if pvc_name:
                        # 备份每个PVC
                        pvc_result = subprocess.run([
                            "kubectl", "get", "pvc", pvc_name, "-n", self.config.namespace,
                            "-o", "yaml"
                        ], capture_output=True, text=True)
                        
                        if pvc_result.returncode == 0:
                            pvc_file = pvc_dir / f"{pvc_name}.yaml"
                            with open(pvc_file, "w", encoding="utf-8") as f:
                                f.write(pvc_result.stdout)
                
                logger.info("✅ 持久化卷备份完成")
                return True
            else:
                logger.error(f"❌ 持久化卷备份失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 持久化卷备份异常: {e}")
            return False
    
    def _backup_application_data(self, backup_path: Path) -> bool:
        """备份应用数据"""
        logger.info("📊 备份应用数据...")
        
        try:
            # 获取Pod列表
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.config.namespace,
                "-l", "app=rqa2025", "-o", "jsonpath={.items[0].metadata.name}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                pod_name = result.stdout.strip()
                data_dir = backup_path / "application_data"
                data_dir.mkdir(exist_ok=True)
                
                # 备份应用数据目录
                backup_result = subprocess.run([
                    "kubectl", "exec", pod_name, "-n", self.config.namespace,
                    "--", "tar", "czf", "-", "/app/data", "/app/models", "/app/logs"
                ], capture_output=True)
                
                if backup_result.returncode == 0:
                    data_file = data_dir / "application_data.tar.gz"
                    with open(data_file, "wb") as f:
                        f.write(backup_result.stdout)
                    logger.info("✅ 应用数据备份完成")
                    return True
                else:
                    logger.warning("⚠️ 应用数据备份跳过（Pod可能未运行）")
                    return True  # 不阻止其他备份
            else:
                logger.warning("⚠️ 未找到运行中的Pod，跳过应用数据备份")
                return True
                
        except Exception as e:
            logger.error(f"❌ 应用数据备份异常: {e}")
            return False
    
    def _create_backup_metadata(self, backup_path: Path, timestamp: str):
        """创建备份元数据"""
        metadata = {
            "backup_time": timestamp,
            "namespace": self.config.namespace,
            "backup_type": "full",
            "compression": self.config.compression,
            "encryption": self.config.encryption,
            "components": [
                "kubernetes_resources",
                "configmaps",
                "secrets",
                "persistent_volumes",
                "application_data"
            ]
        }
        
        metadata_file = backup_path / "backup_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _compress_backup(self, backup_path: Path):
        """压缩备份"""
        logger.info("🗜️ 压缩备份...")
        
        try:
            archive_name = f"{backup_path.name}.tar.gz"
            archive_path = backup_path.parent / archive_name
            
            result = subprocess.run([
                "tar", "-czf", str(archive_path), "-C", str(backup_path.parent), backup_path.name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # 删除原始目录
                shutil.rmtree(backup_path)
                logger.info(f"✅ 备份压缩完成: {archive_path}")
            else:
                logger.error(f"❌ 备份压缩失败: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ 备份压缩异常: {e}")
    
    def _encrypt_backup(self, backup_path: Path):
        """加密备份"""
        logger.info("🔒 加密备份...")
        
        try:
            # 使用gpg加密（如果可用）
            result = subprocess.run([
                "gpg", "--version"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # 使用gpg加密
                if backup_path.is_file():
                    gpg_result = subprocess.run([
                        "gpg", "--encrypt", "--recipient", "rqa2025@example.com",
                        str(backup_path)
                    ], capture_output=True, text=True)
                    
                    if gpg_result.returncode == 0:
                        logger.info("✅ 备份加密完成")
                    else:
                        logger.warning("⚠️ 备份加密失败，使用未加密备份")
                else:
                    logger.warning("⚠️ 备份文件不存在，跳过加密")
            else:
                logger.warning("⚠️ GPG不可用，使用未加密备份")
                
        except Exception as e:
            logger.error(f"❌ 备份加密异常: {e}")
    
    def restore_backup(self, backup_name: str) -> bool:
        """恢复备份"""
        logger.info(f"🔄 开始恢复备份: {backup_name}")
        
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.error(f"❌ 备份不存在: {backup_name}")
                return False
            
            # 1. 验证备份完整性
            if not self._verify_backup(backup_path):
                logger.error("❌ 备份验证失败")
                return False
            
            # 2. 恢复Kubernetes资源
            if not self._restore_k8s_resources(backup_path):
                logger.error("❌ Kubernetes资源恢复失败")
                return False
            
            # 3. 恢复配置映射
            if not self._restore_configmaps(backup_path):
                logger.error("❌ 配置映射恢复失败")
                return False
            
            # 4. 恢复密钥
            if not self._restore_secrets(backup_path):
                logger.error("❌ 密钥恢复失败")
                return False
            
            # 5. 恢复持久化卷
            if not self._restore_persistent_volumes(backup_path):
                logger.error("❌ 持久化卷恢复失败")
                return False
            
            # 6. 恢复应用数据
            if not self._restore_application_data(backup_path):
                logger.error("❌ 应用数据恢复失败")
                return False
            
            logger.info(f"✅ 备份恢复成功: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 备份恢复失败: {e}")
            return False
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """验证备份完整性"""
        logger.info("🔍 验证备份完整性...")
        
        try:
            # 检查必要文件
            required_files = [
                "kubernetes_resources.yaml",
                "configmaps.yaml",
                "secrets.yaml",
                "backup_metadata.json"
            ]
            
            for file_name in required_files:
                file_path = backup_path / file_name
                if not file_path.exists():
                    logger.error(f"❌ 缺少必要文件: {file_name}")
                    return False
            
            logger.info("✅ 备份完整性验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 备份验证异常: {e}")
            return False
    
    def _restore_k8s_resources(self, backup_path: Path) -> bool:
        """恢复Kubernetes资源"""
        logger.info("📋 恢复Kubernetes资源...")
        
        try:
            k8s_file = backup_path / "kubernetes_resources.yaml"
            if k8s_file.exists():
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(k8s_file), "-n", self.config.namespace
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Kubernetes资源恢复完成")
                    return True
                else:
                    logger.error(f"❌ Kubernetes资源恢复失败: {result.stderr}")
                    return False
            else:
                logger.warning("⚠️ Kubernetes资源文件不存在，跳过恢复")
                return True
                
        except Exception as e:
            logger.error(f"❌ Kubernetes资源恢复异常: {e}")
            return False
    
    def _restore_configmaps(self, backup_path: Path) -> bool:
        """恢复配置映射"""
        logger.info("⚙️ 恢复配置映射...")
        
        try:
            config_file = backup_path / "configmaps.yaml"
            if config_file.exists():
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(config_file), "-n", self.config.namespace
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ 配置映射恢复完成")
                    return True
                else:
                    logger.error(f"❌ 配置映射恢复失败: {result.stderr}")
                    return False
            else:
                logger.warning("⚠️ 配置映射文件不存在，跳过恢复")
                return True
                
        except Exception as e:
            logger.error(f"❌ 配置映射恢复异常: {e}")
            return False
    
    def _restore_secrets(self, backup_path: Path) -> bool:
        """恢复密钥"""
        logger.info("🔐 恢复密钥...")
        
        try:
            secrets_file = backup_path / "secrets.yaml"
            if secrets_file.exists():
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(secrets_file), "-n", self.config.namespace
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ 密钥恢复完成")
                    return True
                else:
                    logger.error(f"❌ 密钥恢复失败: {result.stderr}")
                    return False
            else:
                logger.warning("⚠️ 密钥文件不存在，跳过恢复")
                return True
                
        except Exception as e:
            logger.error(f"❌ 密钥恢复异常: {e}")
            return False
    
    def _restore_persistent_volumes(self, backup_path: Path) -> bool:
        """恢复持久化卷"""
        logger.info("💾 恢复持久化卷...")
        
        try:
            pvc_dir = backup_path / "persistent_volumes"
            if pvc_dir.exists():
                for pvc_file in pvc_dir.glob("*.yaml"):
                    result = subprocess.run([
                        "kubectl", "apply", "-f", str(pvc_file), "-n", self.config.namespace
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.warning(f"⚠️ PVC恢复失败: {pvc_file.name}")
                
                logger.info("✅ 持久化卷恢复完成")
                return True
            else:
                logger.warning("⚠️ 持久化卷目录不存在，跳过恢复")
                return True
                
        except Exception as e:
            logger.error(f"❌ 持久化卷恢复异常: {e}")
            return False
    
    def _restore_application_data(self, backup_path: Path) -> bool:
        """恢复应用数据"""
        logger.info("📊 恢复应用数据...")
        
        try:
            data_dir = backup_path / "application_data"
            data_file = data_dir / "application_data.tar.gz"
            
            if data_file.exists():
                # 获取Pod名称
                result = subprocess.run([
                    "kubectl", "get", "pods", "-n", self.config.namespace,
                    "-l", "app=rqa2025", "-o", "jsonpath={.items[0].metadata.name}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    pod_name = result.stdout.strip()
                    
                    # 恢复应用数据
                    with open(data_file, "rb") as f:
                        restore_result = subprocess.run([
                            "kubectl", "exec", pod_name, "-n", self.config.namespace,
                            "--", "tar", "xzf", "-", "-C", "/"
                        ], input=f.read(), capture_output=True)
                    
                    if restore_result.returncode == 0:
                        logger.info("✅ 应用数据恢复完成")
                        return True
                    else:
                        logger.error(f"❌ 应用数据恢复失败: {restore_result.stderr}")
                        return False
                else:
                    logger.warning("⚠️ 未找到运行中的Pod，跳过应用数据恢复")
                    return True
            else:
                logger.warning("⚠️ 应用数据文件不存在，跳过恢复")
                return True
                
        except Exception as e:
            logger.error(f"❌ 应用数据恢复异常: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        logger.info("📋 列出备份...")
        
        backups = []
        try:
            for item in self.backup_dir.iterdir():
                if item.is_dir() and item.name.startswith("backup_"):
                    metadata_file = item / "backup_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        backups.append({
                            "name": item.name,
                            "time": metadata.get("backup_time", "unknown"),
                            "type": metadata.get("backup_type", "unknown"),
                            "size": self._get_directory_size(item)
                        })
            
            # 按时间排序
            backups.sort(key=lambda x: x["time"], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"❌ 列出备份失败: {e}")
            return []
    
    def _get_directory_size(self, path: Path) -> str:
        """获取目录大小"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            # 转换为人类可读格式
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
            
        except Exception:
            return "unknown"
    
    def cleanup_old_backups(self) -> bool:
        """清理旧备份"""
        logger.info("🧹 清理旧备份...")
        
        try:
            backups = self.list_backups()
            current_time = datetime.now()
            
            # 删除超过保留期的备份
            for backup in backups:
                try:
                    backup_time = datetime.strptime(backup["time"], "%Y%m%d_%H%M%S")
                    age_days = (current_time - backup_time).days
                    
                    if age_days > self.config.retention_days:
                        backup_path = self.backup_dir / backup["name"]
                        if backup_path.exists():
                            shutil.rmtree(backup_path)
                            logger.info(f"🗑️ 删除旧备份: {backup['name']}")
                except Exception as e:
                    logger.warning(f"⚠️ 处理备份时出错: {backup['name']}, {e}")
            
            # 如果备份数量超过限制，删除最旧的
            if len(backups) > self.config.max_backups:
                backups_to_delete = backups[self.config.max_backups:]
                for backup in backups_to_delete:
                    backup_path = self.backup_dir / backup["name"]
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                        logger.info(f"🗑️ 删除超量备份: {backup['name']}")
            
            logger.info("✅ 旧备份清理完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 清理旧备份失败: {e}")
            return False
    
    def schedule_backup(self, schedule_time: str = "02:00"):
        """调度定时备份"""
        logger.info(f"⏰ 设置定时备份: {schedule_time}")
        
        def backup_job():
            logger.info("🔄 执行定时备份...")
            if self.create_backup():
                logger.info("✅ 定时备份完成")
                # 清理旧备份
                self.cleanup_old_backups()
            else:
                logger.error("❌ 定时备份失败")
        
        # 设置定时任务
        schedule.every().day.at(schedule_time).do(backup_job)
        
        logger.info(f"✅ 定时备份已设置: 每天 {schedule_time}")
        
        # 运行调度器
        while True:
            schedule.run_pending()
            time.sleep(60)

def main():
    """主函数"""
    print("💾 RQA2025 备份管理工具")
    print("=" * 50)
    
    # 创建备份配置
    config = BackupConfig()
    
    # 创建备份管理器
    backup_manager = BackupManager(config)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            success = backup_manager.create_backup()
            if success:
                print("✅ 备份创建成功")
            else:
                print("❌ 备份创建失败")
                sys.exit(1)
        
        elif command == "restore":
            if len(sys.argv) > 2:
                backup_name = sys.argv[2]
                success = backup_manager.restore_backup(backup_name)
                if success:
                    print("✅ 备份恢复成功")
                else:
                    print("❌ 备份恢复失败")
                    sys.exit(1)
            else:
                print("❌ 请指定要恢复的备份名称")
                sys.exit(1)
        
        elif command == "list":
            backups = backup_manager.list_backups()
            if backups:
                print("📋 可用备份:")
                for backup in backups:
                    print(f"  - {backup['name']} ({backup['time']}) - {backup['size']}")
            else:
                print("📋 没有找到备份")
        
        elif command == "cleanup":
            success = backup_manager.cleanup_old_backups()
            if success:
                print("✅ 旧备份清理完成")
            else:
                print("❌ 旧备份清理失败")
                sys.exit(1)
        
        elif command == "schedule":
            schedule_time = sys.argv[2] if len(sys.argv) > 2 else "02:00"
            print(f"⏰ 启动定时备份调度器 (每天 {schedule_time})")
            backup_manager.schedule_backup(schedule_time)
        
        else:
            print("❌ 未知命令")
            print("可用命令: create, restore <backup_name>, list, cleanup, schedule [time]")
            sys.exit(1)
    else:
        print("用法: python backup_manager.py <command> [options]")
        print("命令:")
        print("  create     - 创建新备份")
        print("  restore    - 恢复指定备份")
        print("  list       - 列出所有备份")
        print("  cleanup    - 清理旧备份")
        print("  schedule   - 启动定时备份调度器")
        sys.exit(1)

if __name__ == "__main__":
    main() 