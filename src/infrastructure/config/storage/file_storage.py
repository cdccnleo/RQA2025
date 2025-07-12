import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from ..interfaces.version_storage import IVersionStorage

class FileVersionStorage(IVersionStorage):
    """文件系统存储适配器实现"""

    def __init__(self, storage_path: str = "versions"):
        self.storage_path = Path(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

    def save_version(self, env: str, version_data: Dict[str, Any]) -> bool:
        """保存版本到文件"""
        env_dir = self.storage_path / env
        os.makedirs(env_dir, exist_ok=True)

        version_file = env_dir / f"{version_data['version']}.json"
        try:
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError):
            return False

    def load_version(self, env: str, version_id: str) -> Optional[Dict[str, Any]]:
        """从文件加载版本"""
        version_file = self.storage_path / env / f"{version_id}.json"
        if not version_file.exists():
            return None

        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def list_versions(self, env: str) -> list:
        """列出环境的所有版本"""
        env_dir = self.storage_path / env
        if not env_dir.exists():
            return []

        versions = []
        for version_file in env_dir.glob("*.json"):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    versions.append({
                        'id': data.get('version', version_file.stem),
                        'timestamp': data.get('timestamp', 0),
                        'author': data.get('author', 'unknown')
                    })
            except (json.JSONDecodeError, IOError):
                continue

        return sorted(versions, key=lambda v: v["timestamp"], reverse=True)
    
    def delete_version(self, env: str, version_id: str) -> bool:
        """删除指定版本
        
        Args:
            env: 环境名称
            version_id: 版本ID
            
        Returns:
            是否删除成功
        """
        version_file = self.storage_path / env / f"{version_id}.json"
        if not version_file.exists():
            return False
            
        try:
            version_file.unlink()
            return True
        except (IOError, OSError):
            return False
    
    def get_version(self, env: str, version_id: str) -> Optional[Dict[str, Any]]:
        """获取指定版本（load_version的别名）
        
        Args:
            env: 环境名称
            version_id: 版本ID
            
        Returns:
            版本数据或None
        """
        return self.load_version(env, version_id)
    
    def save(self, file_path: str, data: Dict[str, Any]) -> bool:
        """保存数据到文件
        
        Args:
            file_path: 文件路径
            data: 要保存的数据
            
        Returns:
            是否保存成功
        """
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError):
            return False
