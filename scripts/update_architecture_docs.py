#!/usr/bin/env python3
"""
架构文档自动更新系统

定期检查代码变更并自动更新架构文档
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


class ArchitectureDocUpdater:
    """架构文档更新器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "docs" / "architecture"
        self.state_file = self.docs_dir / ".doc_state.json"

    def check_for_updates(self) -> bool:
        """检查是否有需要更新的内容"""
        print("🔍 检查代码变更...")

        # 获取上次更新时间
        last_update = self._get_last_update_time()

        # 检查源代码变更
        src_changes = self._check_src_changes(last_update)
        config_changes = self._check_config_changes(last_update)

        if src_changes or config_changes:
            print("📝 发现变更，需要更新文档")
            return True
        else:
            print("✅ 代码无变更，无需更新")
            return False

    def update_documentation(self):
        """更新架构文档"""
        print("🔄 开始更新架构文档...")

        # 生成新文档
        result = subprocess.run([
            "python", "scripts/generate_architecture_docs.py"
        ], capture_output=True, text=True, cwd=self.project_root)

        if result.returncode == 0:
            print("✅ 文档生成成功")

            # 更新状态文件
            self._update_state()

            # 提交更改
            self._commit_changes()

        else:
            print(f"❌ 文档生成失败: {result.stderr}")

    def _get_last_update_time(self) -> datetime:
        """获取上次更新时间"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return datetime.fromisoformat(state.get('last_update', '2025-01-01T00:00:00'))
            except:
                pass

        return datetime.now() - timedelta(days=1)

    def _check_src_changes(self, since: datetime) -> bool:
        """检查源代码变更"""
        try:
            # 使用git检查变更
            result = subprocess.run([
                "git", "log", "--since", since.isoformat(),
                "--name-only", "--pretty=format:", "src/"
            ], capture_output=True, text=True, cwd=self.project_root)

            return bool(result.stdout.strip())
        except:
            # 如果git不可用，检查文件修改时间
            src_dir = self.project_root / "src"
            for file_path in src_dir.rglob("*.py"):
                if file_path.stat().st_mtime > since.timestamp():
                    return True
            return False

    def _check_config_changes(self, since: datetime) -> bool:
        """检查配置文件变更"""
        config_dir = self.project_root / "config"
        if not config_dir.exists():
            return False

        for file_path in config_dir.rglob("*"):
            if file_path.stat().st_mtime > since.timestamp():
                return True
        return False

    def _update_state(self):
        """更新状态文件"""
        state = {
            "last_update": datetime.now().isoformat(),
            "version": "auto_generated"
        }

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _commit_changes(self):
        """提交文档变更"""
        try:
            # 添加更改
            subprocess.run(["git", "add", "docs/architecture/"], cwd=self.project_root)

            # 检查是否有更改
            result = subprocess.run(["git", "status", "--porcelain"],
                                    cwd=self.project_root, capture_output=True, text=True)

            if result.stdout.strip():
                # 提交更改
                commit_msg = f"docs: 自动更新架构文档 [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=self.project_root)
                print("✅ 文档变更已提交")
            else:
                print("ℹ️  没有需要提交的变更")

        except Exception as e:
            print(f"⚠️  提交失败: {e}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    updater = ArchitectureDocUpdater(project_root)

    if updater.check_for_updates():
        updater.update_documentation()
    else:
        print("📋 文档已是最新状态")


if __name__ == "__main__":
    main()
