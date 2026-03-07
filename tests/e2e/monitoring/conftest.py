"""
监控界面端到端测试的共享配置和fixture

提供所有监控界面测试共用的fixture和工具函数。
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session")
def web_static_dir(project_root):
    """获取web-static目录路径"""
    return project_root / "web-static"


def read_html_file(file_path):
    """读取HTML文件的辅助函数"""
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def assert_html_structure(content, min_components=3):
    """
    验证HTML基本结构的辅助函数
    
    Args:
        content: HTML内容
        min_components: 最少应找到的组件数量
    """
    components = {
        "DOCTYPE": "<!DOCTYPE html>" in content,
        "html_tag": "<html" in content,
        "head": "<head>" in content,
        "body": "<body>" in content,
    }
    
    found = sum(1 for v in components.values() if v)
    assert found >= min_components, f"HTML结构不完整，仅找到 {found}/{len(components)} 个必需组件"
    
    return components

