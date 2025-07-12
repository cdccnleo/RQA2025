import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import RLock
from dataclasses import asdict
from .interfaces.version_storage import IVersionStorage
from .config_version import ConfigVersion

class FileVersionStorage(IVersionStorage):
    """文件系统版本存储实现"""

    def __init__(self, storage_path: str = "versions"):
        self.storage_path = Path(storage_path)
        self._lock = RLock()
        os.makedirs(self.storage_path, exist_ok=True)

    def save_version(self, env: str, version_data: Dict[str, Any]) -> bool:
        """保存版本数据"""
        version = ConfigVersion(**version_data)
        version_file = self._get_version_file(env, version.version)

        with self._lock:
            try:
                with open(version_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(version), f, ensure_ascii=False, indent=2)
                return True
            except (IOError, TypeError) as e:
                return False

    def load_version(self, env: str, version_id: str) -> Optional[Dict[str, Any]]:
        """加载版本数据"""
        version_file = self._get_version_file(env, version_id)
        if not version_file.exists():
            return None

        with self._lock:
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
            except (json.JSONDecodeError, IOError):
                return None

    def list_versions(self, env: str) -> List[Dict[str, Any]]:
        """列出所有版本"""
        env_dir = self.storage_path / env
        if not env_dir.exists():
            return []

        versions = []
        with self._lock:
            for version_file in env_dir.glob("*.json"):
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        versions.append({
                            'version': data.get('version', version_file.stem),
                            'timestamp': data.get('timestamp', 0),
                            'checksum': data.get('checksum', ''),
                            'author': data.get('author', 'unknown'),
                            'comment': data.get('comment', '')
                        })
                except (json.JSONDecodeError, IOError):
                    continue

        return sorted(versions, key=lambda v: v["timestamp"], reverse=True)

    def _get_version_file(self, env: str, version_id: str) -> Path:
        """获取版本文件路径"""
        version_dir = self.storage_path / env
        os.makedirs(version_dir, exist_ok=True)
        return version_dir / f"{version_id}.json"
