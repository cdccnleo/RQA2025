#!/usr/bin/env python3
"""
RQA2025 版本管理规范化工具

提供完善的文档版本管理功能，包括：
- 版本号自动生成和管理
- 版本历史追踪
- 版本发布流程管理
- 版本兼容性检查
- 版本分支管理
"""

import os
import re
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import semver

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VersionManager:
    """版本管理器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.version_file = self.project_root / "VERSION"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        self.version_history_file = self.project_root / "version_history.json"

        # 版本模式
        self.version_patterns = {
            'document': r'v(\d+)\.(\d+)\.(\d+)',  # 文档版本
            'code': r'__version__\s*=\s*["\']([^"\']+)["\']',  # 代码版本
            'api': r'v(\d+)\.(\d+)\.(\d+)',  # API版本
        }

    def get_current_version(self) -> Dict[str, str]:
        """获取当前版本信息"""
        versions = {}

        # 1. 获取主版本号
        if self.version_file.exists():
            versions['main'] = self.version_file.read_text().strip()

        # 2. 获取文档版本
        doc_versions = self._scan_document_versions()
        versions['documents'] = doc_versions

        # 3. 获取代码版本
        code_versions = self._scan_code_versions()
        versions['code'] = code_versions

        return versions

    def bump_version(self, bump_type: str = 'patch', scope: str = 'all') -> Dict[str, Any]:
        """版本号递增"""
        result = {
            'success': False,
            'old_version': None,
            'new_version': None,
            'changes': []
        }

        try:
            # 获取当前版本
            current_versions = self.get_current_version()
            current_version = current_versions.get('main', '0.0.0')

            # 解析当前版本
            if not semver.VersionInfo.isvalid(current_version):
                current_version = '0.0.0'

            # 递增版本
            if bump_type == 'major':
                new_version = semver.bump_major(current_version)
            elif bump_type == 'minor':
                new_version = semver.bump_minor(current_version)
            elif bump_type == 'patch':
                new_version = semver.bump_patch(current_version)
            else:
                raise ValueError(f"不支持的版本递增类型: {bump_type}")

            # 更新版本文件
            self.version_file.write_text(new_version)

            # 更新指定范围的版本
            if scope == 'all' or scope == 'documents':
                self._update_document_versions(current_version, new_version)

            if scope == 'all' or scope == 'code':
                self._update_code_versions(current_version, new_version)

            # 记录版本历史
            self._record_version_change(current_version, new_version, bump_type)

            result.update({
                'success': True,
                'old_version': current_version,
                'new_version': new_version,
                'bump_type': bump_type,
                'scope': scope
            })

            logger.info(f"版本已递增: {current_version} -> {new_version} ({bump_type})")

        except Exception as e:
            logger.error(f"版本递增失败: {e}")
            result['error'] = str(e)

        return result

    def create_release(self, version: str = None, release_notes: str = None) -> Dict[str, Any]:
        """创建版本发布"""
        result = {
            'success': False,
            'version': version,
            'release_date': datetime.now().isoformat(),
            'changes': []
        }

        try:
            # 如果没有指定版本，使用当前版本
            if not version:
                current = self.get_current_version()
                version = current.get('main', '0.0.0')

            # 验证版本格式
            if not semver.VersionInfo.isvalid(version):
                raise ValueError(f"无效的版本格式: {version}")

            # 更新CHANGELOG
            if release_notes:
                self._update_changelog(version, release_notes)

            # 创建git标签
            self._create_git_tag(version, release_notes)

            # 生成发布说明
            release_notes_content = self._generate_release_notes(version)

            result.update({
                'success': True,
                'release_notes': release_notes_content
            })

            logger.info(f"版本发布创建成功: {version}")

        except Exception as e:
            logger.error(f"创建版本发布失败: {e}")
            result['error'] = str(e)

        return result

    def check_version_consistency(self) -> Dict[str, Any]:
        """检查版本一致性"""
        result = {
            'consistent': True,
            'issues': [],
            'recommendations': []
        }

        try:
            versions = self.get_current_version()

            # 检查主版本与文档版本一致性
            main_version = versions.get('main')
            if main_version:
                doc_versions = versions.get('documents', {})
                inconsistent_docs = []

                for doc, doc_version in doc_versions.items():
                    if doc_version != main_version:
                        inconsistent_docs.append({
                            'document': doc,
                            'doc_version': doc_version,
                            'main_version': main_version
                        })

                if inconsistent_docs:
                    result['consistent'] = False
                    result['issues'].extend(inconsistent_docs)
                    result['recommendations'].append("建议同步文档版本与主版本一致")

            # 检查代码版本一致性
            code_versions = versions.get('code', {})
            if len(set(code_versions.values())) > 1:
                result['consistent'] = False
                result['issues'].append({
                    'type': 'code_version_inconsistency',
                    'versions': code_versions
                })
                result['recommendations'].append("建议统一所有代码模块的版本号")

        except Exception as e:
            logger.error(f"版本一致性检查失败: {e}")
            result['error'] = str(e)

        return result

    def generate_version_report(self) -> Dict[str, Any]:
        """生成版本报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'current_versions': self.get_current_version(),
            'consistency_check': self.check_version_consistency(),
            'recent_changes': self._get_recent_changes(),
            'upcoming_releases': self._get_upcoming_releases()
        }

        # 保存报告
        report_file = self.project_root / "reports" / "version_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"版本报告已生成: {report_file}")
        return report

    def _scan_document_versions(self) -> Dict[str, str]:
        """扫描文档版本"""
        versions = {}

        # 扫描架构文档
        docs_dir = self.project_root / "docs" / "architecture"
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                try:
                    content = doc_file.read_text(encoding='utf-8')
                    match = re.search(self.version_patterns['document'], content)
                    if match:
                        versions[doc_file.name] = match.group(0)
                except Exception as e:
                    logger.warning(f"扫描文档版本失败 {doc_file}: {e}")

        return versions

    def _scan_code_versions(self) -> Dict[str, str]:
        """扫描代码版本"""
        versions = {}

        # 扫描Python文件
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                match = re.search(self.version_patterns['code'], content)
                if match:
                    versions[str(py_file.relative_to(self.project_root))] = match.group(1)
            except Exception as e:
                logger.warning(f"扫描代码版本失败 {py_file}: {e}")

        return versions

    def _update_document_versions(self, old_version: str, new_version: str):
        """更新文档版本"""
        docs_dir = self.project_root / "docs" / "architecture"

        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                try:
                    content = doc_file.read_text(encoding='utf-8')
                    # 替换版本号
                    updated_content = re.sub(
                        self.version_patterns['document'],
                        f"v{new_version}",
                        content
                    )
                    doc_file.write_text(updated_content, encoding='utf-8')
                    logger.info(f"文档版本已更新: {doc_file.name}")
                except Exception as e:
                    logger.error(f"更新文档版本失败 {doc_file}: {e}")

    def _update_code_versions(self, old_version: str, new_version: str):
        """更新代码版本"""
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                # 替换版本号
                updated_content = re.sub(
                    self.version_patterns['code'],
                    f'__version__ = "{new_version}"',
                    content
                )
                py_file.write_text(updated_content, encoding='utf-8')
            except Exception as e:
                logger.warning(f"更新代码版本失败 {py_file}: {e}")

    def _record_version_change(self, old_version: str, new_version: str, bump_type: str):
        """记录版本变更"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'old_version': old_version,
            'new_version': new_version,
            'bump_type': bump_type,
            'changes': self._get_recent_changes()
        }

        # 读取现有历史
        history = []
        if self.version_history_file.exists():
            try:
                with open(self.version_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        # 添加新记录
        history.append(history_entry)

        # 保存历史
        with open(self.version_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _update_changelog(self, version: str, notes: str):
        """更新CHANGELOG"""
        if not self.changelog_file.exists():
            # 创建新的CHANGELOG
            content = f"# Changelog\n\n"
        else:
            content = self.changelog_file.read_text(encoding='utf-8')

        # 添加新版本条目
        today = datetime.now().strftime("%Y-%m-%d")
        version_entry = f"\n## [{version}] - {today}\n\n{notes}\n"

        # 在文件开头插入
        if content.startswith("# Changelog"):
            content = content.replace("# Changelog", f"# Changelog{version_entry}", 1)
        else:
            content = f"# Changelog{version_entry}\n{content}"

        self.changelog_file.write_text(content, encoding='utf-8')

    def _create_git_tag(self, version: str, message: str = None):
        """创建Git标签"""
        try:
            # 创建标签
            cmd = ["git", "tag", "-a", f"v{version}", "-m", message or f"Release version {version}"]
            subprocess.run(cmd, cwd=self.project_root, check=True)

            # 推送标签
            subprocess.run(
                ["git", "push", "origin", f"v{version}"], cwd=self.project_root, check=True)

            logger.info(f"Git标签已创建: v{version}")

        except subprocess.CalledProcessError as e:
            logger.warning(f"创建Git标签失败: {e}")
        except FileNotFoundError:
            logger.warning("Git命令不可用，跳过标签创建")

    def _generate_release_notes(self, version: str) -> str:
        """生成发布说明"""
        notes = f"# Release Notes - v{version}\n\n"
        notes += f"**发布日期**: {datetime.now().strftime('%Y年%m月%d日')}\n\n"

        # 从CHANGELOG获取变更内容
        if self.changelog_file.exists():
            content = self.changelog_file.read_text(encoding='utf-8')
            # 提取当前版本的变更
            version_pattern = f"## \\[{version}\\](.*?)(\\n## \\[|\\Z)"
            match = re.search(version_pattern, content, re.DOTALL)
            if match:
                notes += "## 主要变更\n\n"
                notes += match.group(1).strip() + "\n\n"

        # 添加版本信息
        notes += "## 版本信息\n\n"
        versions = self.get_current_version()
        notes += f"- **主版本**: {versions.get('main', 'N/A')}\n"
        notes += f"- **文档版本**: {len(versions.get('documents', {}))} 个文档\n"
        notes += f"- **代码模块**: {len(versions.get('code', {}))} 个模块\n\n"

        return notes

    def _get_recent_changes(self) -> List[str]:
        """获取最近的变更"""
        changes = []

        try:
            # 获取git提交历史
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                changes = result.stdout.strip().split('\n')

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("无法获取Git提交历史")

        return changes

    def _get_upcoming_releases(self) -> List[Dict[str, Any]]:
        """获取即将发布的版本"""
        # 这里可以实现从issue跟踪系统或项目管理工具获取信息
        return [
            {
                'version': '1.1.0',
                'planned_date': '2025-02-01',
                'features': ['新功能1', '新功能2']
            }
        ]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 版本管理工具')
    parser.add_argument('--project-root', help='项目根目录路径')
    parser.add_argument('action', choices=['bump', 'release', 'check', 'report'],
                        help='执行的操作')
    parser.add_argument('--type', choices=['major', 'minor', 'patch'],
                        default='patch', help='版本递增类型')
    parser.add_argument('--scope', choices=['all', 'documents', 'code'],
                        default='all', help='更新范围')
    parser.add_argument('--version', help='指定版本号')
    parser.add_argument('--notes', help='发布说明')

    args = parser.parse_args()

    # 创建版本管理器
    vm = VersionManager(args.project_root)

    if args.action == 'bump':
        # 版本递增
        result = vm.bump_version(args.type, args.scope)
        if result['success']:
            print(f"版本已递增: {result['old_version']} -> {result['new_version']}")
        else:
            print(f"版本递增失败: {result.get('error', '未知错误')}")

    elif args.action == 'release':
        # 创建发布
        result = vm.create_release(args.version, args.notes)
        if result['success']:
            print(f"版本发布创建成功: {result['version']}")
        else:
            print(f"创建版本发布失败: {result.get('error', '未知错误')}")

    elif args.action == 'check':
        # 版本一致性检查
        result = vm.check_version_consistency()
        if result['consistent']:
            print("✅ 版本一致性检查通过")
        else:
            print("❌ 发现版本不一致问题:")
            for issue in result.get('issues', []):
                print(f"  • {issue}")

    elif args.action == 'report':
        # 生成版本报告
        result = vm.generate_version_report()
        print("版本报告已生成")
        print(f"当前主版本: {result['current_versions'].get('main', 'N/A')}")
        print(f"文档数量: {len(result['current_versions'].get('documents', {}))}")
        print(f"代码模块数量: {len(result['current_versions'].get('code', {}))}")


if __name__ == "__main__":
    main()
