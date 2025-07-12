import time
from copy import deepcopy
from typing import Dict, Optional, List, Union
import os
import json
import difflib
from pathlib import Path
from datetime import datetime, timedelta
from src.infrastructure.config.interfaces.version_manager import IVersionManager

class VersionService(IVersionManager):
    """版本控制服务实现"""
    
    def __init__(self, storage_dir: str = "data/versions"):
        self._versions = {}  # 内存中的版本缓存
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_versions()
        
    def _load_persisted_versions(self):
        """从文件系统加载持久化的版本"""
        for env_file in self.storage_dir.glob("*.json"):
            env = env_file.stem
            with open(env_file, 'r', encoding='utf-8') as f:
                self._versions[env] = json.load(f)

    def _persist_versions(self, env: str):
        """将版本数据持久化到文件系统"""
        env_file = self.storage_dir / f"{env}.json"
        with open(env_file, 'w', encoding='utf-8') as f:
            json.dump(self._versions[env], f, indent=2)

    def add_version(self, env: str, config: Dict) -> str:
        """添加新版本"""
        if env not in self._versions:
            self._versions[env] = []

        version_id = f"v{len(self._versions[env]) + 1}"
        version_data = {
            'id': version_id,
            'config': deepcopy(config),
            'timestamp': time.time()
        }
        self._versions[env].append(version_data)
        return version_id

    def cleanup_old_versions(self, env: str, keep_last: int = 10) -> int:
        """清理旧版本，保留最近的指定数量"""
        if env not in self._versions or len(self._versions[env]) <= keep_last:
            return 0
            
        removed = len(self._versions[env]) - keep_last
        self._versions[env] = self._versions[env][-keep_last:]
        self._persist_versions(env)
        return removed

    def get_version(self, env: str, version: Union[int, str]) -> Optional[Dict]:
        """获取特定版本配置

        Args:
            env: 环境名称
            version: 版本号(整数索引或字符串ID)

        Returns:
            配置字典，如果找不到则返回None
        """
        if env not in self._versions or not self._versions[env]:
            return None

        # 如果version是整数，处理索引
        if isinstance(version, int):
            # 处理负索引
            if version < 0:
                version = len(self._versions[env]) + version
            # 检查索引范围
            if version < 0 or version >= len(self._versions[env]):
                return None
            return self._versions[env][version]['config']

        # 如果version是字符串，按版本ID查找
        for v in self._versions[env]:
            if v['id'] == version:
                return v['config']

        return None
        
    def diff_versions(self, env: str, v1: str, v2: str) -> Dict:
        """比较两个版本差异，返回详细差异报告"""
        config1 = self.get_version(env, v1)
        config2 = self.get_version(env, v2)
        
        if not config1 or not config2:
            raise ValueError("版本不存在")

        # 使用difflib进行专业差异比较
        diff = {
            'summary': {
                'added': 0,
                'removed': 0,
                'changed': 0,
                'unchanged': 0
            },
            'details': []
        }

        # 序列化配置为文本行以便比较
        def serialize_config(config):
            lines = []
            for k, v in sorted(config.items()):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            return lines

        lines1 = serialize_config(config1)
        lines2 = serialize_config(config2)

        # 使用difflib计算差异
        differ = difflib.Differ()
        delta = list(differ.compare(lines1, lines2))

        # 解析差异结果
        current_diff = None
        for line in delta:
            if line.startswith('  '):  # 未变化
                if current_diff:
                    diff['details'].append(current_diff)
                    current_diff = None
            elif line.startswith('- '):  # 删除
                diff['summary']['removed'] += 1
                current_diff = {
                    'key': line[2:].split(':', 1)[0],
                    'old': json.loads(line[2:].split(':', 1)[1].strip()),
                    'new': None
                }
            elif line.startswith('+ '):  # 新增
                diff['summary']['added'] += 1
                if current_diff and current_diff['new'] is None:
                    current_diff['new'] = json.loads(line[2:].split(':', 1)[1].strip())
                    diff['summary']['changed'] += 1
                    diff['summary']['removed'] -= 1
                else:
                    if current_diff:
                        diff['details'].append(current_diff)
                    current_diff = {
                        'key': line[2:].split(':', 1)[0],
                        'old': None,
                        'new': json.loads(line[2:].split(':', 1)[1].strip())
                    }
        
        if current_diff:
            diff['details'].append(current_diff)

        return diff

    def rollback(self, env: str, version: str) -> bool:
        """回滚到指定版本"""
        if env not in self._versions or not self._versions[env]:
            return False

        # 查找目标版本索引
        target_index = next(
            (i for i, v in enumerate(self._versions[env])
             if v['id'] == version), None)

        if target_index is None:
            return False

        # 创建回滚版本
        rollback_version = {
            'id': f"rollback_{time.time()}",
            'config': deepcopy(self._versions[env][target_index]['config']),
            'timestamp': time.time(),
            'is_rollback': True,
            'original_version': version
        }

        # 添加到版本列表
        self._versions[env].append(rollback_version)
        return True

    def _publish_event(self, event_type: str, payload: Dict):
        """发布版本控制事件"""
        # 实际实现应该使用事件总线
        # 这里简化处理，只打印日志
        print(f"Event published: {event_type}, Payload: {json.dumps(payload)}")
        
    def _current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
