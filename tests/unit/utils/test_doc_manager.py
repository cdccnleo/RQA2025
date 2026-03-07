"""
工具层 - doc_manager.py 测试

测试src/utils/devtools/doc_manager.py的基本功能
"""

import sys
from pathlib import Path

# 确保Python路径正确配置（必须在所有导入之前）
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

# 确保路径在sys.path的最前面
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
if src_path_str in sys.path:
    sys.path.remove(src_path_str)

sys.path.insert(0, project_root_str)
sys.path.insert(0, src_path_str)

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


def test_document_metadata():
    """测试DocumentMetadata数据类"""
    from src.utils.devtools.doc_manager import DocumentMetadata
    
    metadata = DocumentMetadata(
        doc_id="test_doc_1",
        title="Test Document",
        version="1.0.0",
        author="Test Author",
        description="Test Description"
    )
    
    assert metadata.doc_id == "test_doc_1"
    assert metadata.title == "Test Document"
    assert metadata.version == "1.0.0"
    assert metadata.author == "Test Author"
    assert metadata.description == "Test Description"
    assert metadata.status == "draft"
    assert isinstance(metadata.tags, list)
    assert isinstance(metadata.reviewers, list)


def test_document_version():
    """测试DocumentVersion数据类"""
    from src.utils.devtools.doc_manager import DocumentVersion
    
    version = DocumentVersion(
        version="1.0.0",
        content_hash="abc123",
        created_at=datetime.now(),
        author="Test Author",
        changes="Initial version"
    )
    
    assert version.version == "1.0.0"
    assert version.content_hash == "abc123"
    assert isinstance(version.created_at, datetime)
    assert version.author == "Test Author"


def test_document_manager_init():
    """测试DocumentManager初始化"""
    from src.utils.devtools.doc_manager import DocumentManager
    
    manager = DocumentManager()
    
    assert hasattr(manager, 'documents')
    assert isinstance(manager.documents, dict)


def test_document_manager_create_document():
    """测试create_document方法"""
    from src.utils.devtools.doc_manager import DocumentManager, DocumentMetadata
    import tempfile
    import shutil
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        manager = DocumentManager(docs_dir=temp_dir)
        
        metadata = DocumentMetadata(
            doc_id="test_doc_1",
            title="Test Document",
            version="1.0.0",
            author="Test Author",
            description="Test Description"
        )
        
        doc_id = manager.create_document(metadata, "Test content")
        
        assert doc_id == "test_doc_1"
        assert "test_doc_1" in manager.documents
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_manager_get_document():
    """测试get_document方法"""
    from src.utils.devtools.doc_manager import DocumentManager, DocumentMetadata
    import tempfile
    import shutil
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        manager = DocumentManager(docs_dir=temp_dir)
        
        metadata = DocumentMetadata(
            doc_id="test_doc_1",
            title="Test Document",
            version="1.0.0",
            author="Test Author",
            description="Test Description"
        )
        
        manager.create_document(metadata, "Test content")
        
        doc = manager.get_document("test_doc_1")
        
        assert doc is not None
        assert doc['metadata']['doc_id'] == "test_doc_1"
        assert doc['content'] == "Test content"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_manager_update_document():
    """测试update_document方法"""
    from src.utils.devtools.doc_manager import DocumentManager, DocumentMetadata
    import tempfile
    import shutil
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        manager = DocumentManager(docs_dir=temp_dir)
        
        metadata = DocumentMetadata(
            doc_id="test_doc_1",
            title="Test Document",
            version="1.0.0",
            author="Test Author",
            description="Test Description"
        )
        
        manager.create_document(metadata, "Original content")
        
        updated = manager.update_document("test_doc_1", "Updated content", "Test Author", "Update test")
        
        assert updated is True
        
        doc = manager.get_document("test_doc_1")
        assert doc['content'] == "Updated content"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

