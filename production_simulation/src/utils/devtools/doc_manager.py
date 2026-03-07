#!/usr/bin/env python3
"""
RQA2025 文档管理器
Documentation Manager

管理项目文档的生成、版本控制和发布。
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception as e:
    # 如果导入失败，使用标准logging
    logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:

    """文档元数据"""
    doc_id: str
    title: str
    version: str
    author: str
    description: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "draft"  # draft, review, published, archived
    reviewers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DocumentVersion:

    """文档版本"""
    version: str
    content_hash: str
    created_at: datetime
    author: str
    changes: str
    parent_version: Optional[str] = None


class DocumentManager:

    """
    文档管理器
    负责文档的版本控制、发布和维护
    """

    def __init__(self, docs_dir: str = "docs"):

        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.documents: Dict[str, DocumentMetadata] = {}
        self.versions: Dict[str, List[DocumentVersion]] = {}

        # 文档索引文件
        self.index_file = self.docs_dir / "DOCUMENT_INDEX.json"

        # 加载现有文档
        self._load_document_index()

        logger.info(f"文档管理器已初始化，文档目录: {self.docs_dir}")

    def create_document(self, metadata: DocumentMetadata, content: str) -> str:
        """创建新文档"""
        # 保存文档内容
        content_file = self.docs_dir / f"{metadata.doc_id}.md"
        with open(content_file, 'w', encoding='utf - 8') as f:
            f.write(content)

        # 保存元数据
        self.documents[metadata.doc_id] = metadata
        self.versions[metadata.doc_id] = []

        # 创建初始版本
        initial_version = DocumentVersion(
            version=metadata.version,
            content_hash=self._calculate_content_hash(content),
            created_at=metadata.created_at,
            author=metadata.author,
            changes="Initial creation"
        )
        self.versions[metadata.doc_id].append(initial_version)

        # 更新索引
        self._save_document_index()

        logger.info(f"创建文档: {metadata.doc_id} - {metadata.title}")
        return metadata.doc_id

    def update_document(self, doc_id: str, content: str, author: str, changes: str) -> bool:
        """更新文档"""
        if doc_id not in self.documents:
            return False

        metadata = self.documents[doc_id]

        # 生成新版本号
        current_version = metadata.version
        new_version = self._increment_version(current_version)

        # 保存新内容
        content_file = self.docs_dir / f"{doc_id}.md"
        with open(content_file, 'w', encoding='utf - 8') as f:
            f.write(content)

        # 更新元数据
        metadata.version = new_version
        metadata.updated_at = datetime.now()

        # 创建新版本记录
        new_version_record = DocumentVersion(
            version=new_version,
            content_hash=self._calculate_content_hash(content),
            created_at=datetime.now(),
            author=author,
            changes=changes,
            parent_version=current_version
        )
        self.versions[doc_id].append(new_version_record)

        # 更新索引
        self._save_document_index()

        logger.info(f"更新文档: {doc_id} 到版本 {new_version}")
        return True

    def get_document(self, doc_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取文档"""
        if doc_id not in self.documents:
            return None

        metadata = self.documents[doc_id]

        # 获取指定版本的内容
        if version:
            content_file = self.docs_dir / f"{doc_id}_{version}.md"
        else:
            content_file = self.docs_dir / f"{doc_id}.md"

        if not content_file.exists():
            return None

        with open(content_file, 'r', encoding='utf - 8') as f:
            content = f.read()

        return {
            'metadata': {
                'doc_id': metadata.doc_id,
                'title': metadata.title,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'tags': metadata.tags,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat(),
                'status': metadata.status,
                'reviewers': metadata.reviewers,
                'dependencies': metadata.dependencies
            },
            'content': content,
            'versions': [
                {
                    'version': v.version,
                    'created_at': v.created_at.isoformat(),
                    'author': v.author,
                    'changes': v.changes
                }
                for v in self.versions.get(doc_id, [])
            ]
        }

    def list_documents(self, status: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """列出文档"""
        documents = []

        for doc_id, metadata in self.documents.items():
            if status and metadata.status != status:
                continue

            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue

            documents.append({
                'doc_id': doc_id,
                'title': metadata.title,
                'version': metadata.version,
                'author': metadata.author,
                'status': metadata.status,
                'tags': metadata.tags,
                'updated_at': metadata.updated_at.isoformat()
            })

        return documents

    def publish_document(self, doc_id: str, reviewer: str) -> bool:
        """发布文档"""
        if doc_id not in self.documents:
            return False

        metadata = self.documents[doc_id]
        metadata.status = "published"
        metadata.reviewers.append(reviewer)
        metadata.updated_at = datetime.now()

        # 更新索引
        self._save_document_index()

        logger.info(f"发布文档: {doc_id}")
        return True

    def archive_document(self, doc_id: str) -> bool:
        """归档文档"""
        if doc_id not in self.documents:
            return False

        metadata = self.documents[doc_id]
        metadata.status = "archived"
        metadata.updated_at = datetime.now()

        # 更新索引
        self._save_document_index()

        logger.info(f"归档文档: {doc_id}")
        return True

    def generate_api_docs(self, module_path: str, output_path: str) -> bool:
        """生成API文档"""
        try:
            # 这里可以集成Sphinx或其他文档生成工具
            # 暂时创建一个简单的API文档结构

            api_doc = f"""# {module_path} API Documentation

Generated at: {datetime.now().isoformat()}

# # Overview

This document contains the API documentation for {module_path}.

# # Classes

# # Functions

# # Constants

---
Generated by RQA2025 Document Manager
"""

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf - 8') as f:
                f.write(api_doc)

            logger.info(f"生成API文档: {output_path}")
            return True

        except Exception as e:
            logger.error(f"生成API文档失败: {str(e)}")
            return False

    def generate_code_docs(self, source_dir: str, output_dir: str) -> bool:
        """生成代码文档"""
        try:
            # 这里可以集成pdoc、mkdocs或其他工具
            # 暂时创建一个简单的代码文档结构

            code_doc = f"""# Code Documentation

Generated at: {datetime.now().isoformat()}

# # Source Directory: {source_dir}

# # Modules

# # Classes

# # Functions

---
Generated by RQA2025 Document Manager
"""

            output_file = Path(output_dir) / "code_docs.md"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf - 8') as f:
                f.write(code_doc)

            logger.info(f"生成代码文档: {output_dir}")
            return True

        except Exception as e:
            logger.error(f"生成代码文档失败: {str(e)}")
            return False

    def _calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        import hashlib
        return hashlib.md5(content.encode('utf - 8')).hexdigest()

    def _increment_version(self, current_version: str) -> str:
        """递增版本号"""
        # 简单的版本递增逻辑
        parts = current_version.split('.')
        if len(parts) >= 3:
            major, minor, patch = parts[:3]
            patch = str(int(patch) + 1)
            return f"{major}.{minor}.{patch}"
        else:
            return f"{current_version}.1"

    def _load_document_index(self):
        """加载文档索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf - 8') as f:
                    index_data = json.load(f)

                # 恢复文档元数据
                for doc_data in index_data.get('documents', []):
                    metadata = DocumentMetadata(
                        doc_id=doc_data['doc_id'],
                        title=doc_data['title'],
                        version=doc_data['version'],
                        author=doc_data['author'],
                        description=doc_data['description'],
                        tags=doc_data.get('tags', []),
                        created_at=datetime.fromisoformat(doc_data['created_at']),
                        updated_at=datetime.fromisoformat(doc_data['updated_at']),
                        status=doc_data.get('status', 'draft'),
                        reviewers=doc_data.get('reviewers', []),
                        dependencies=doc_data.get('dependencies', [])
                    )
                    self.documents[metadata.doc_id] = metadata

                # 恢复版本信息
                for doc_id, versions_data in index_data.get('versions', {}).items():
                    versions = []
                    for v_data in versions_data:
                        version = DocumentVersion(
                            version=v_data['version'],
                            content_hash=v_data['content_hash'],
                            created_at=datetime.fromisoformat(v_data['created_at']),
                            author=v_data['author'],
                            changes=v_data['changes'],
                            parent_version=v_data.get('parent_version')
                        )
                        versions.append(version)
                    self.versions[doc_id] = versions

                logger.info(f"加载了 {len(self.documents)} 个文档索引")

            except Exception as e:
                logger.error(f"加载文档索引失败: {str(e)}")

    def _save_document_index(self):
        """保存文档索引"""
        index_data = {
            'documents': [
                {
                    'doc_id': metadata.doc_id,
                    'title': metadata.title,
                    'version': metadata.version,
                    'author': metadata.author,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'status': metadata.status,
                    'reviewers': metadata.reviewers,
                    'dependencies': metadata.dependencies
                }
                for metadata in self.documents.values()
            ],
            'versions': {
                doc_id: [
                    {
                        'version': v.version,
                        'content_hash': v.content_hash,
                        'created_at': v.created_at.isoformat(),
                        'author': v.author,
                        'changes': v.changes,
                        'parent_version': v.parent_version
                    }
                    for v in versions
                ]
                for doc_id, versions in self.versions.items()
            },
            'last_updated': datetime.now().isoformat()
        }

        with open(self.index_file, 'w', encoding='utf - 8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)


# 创建全局文档管理器实例
_doc_manager = None


def get_document_manager() -> DocumentManager:
    """获取全局文档管理器实例"""
    global _doc_manager
    if _doc_manager is None:
        _doc_manager = DocumentManager()
    return _doc_manager


__all__ = [
    'DocumentManager', 'DocumentMetadata', 'DocumentVersion', 'get_document_manager'
]
