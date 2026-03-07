"""
文档加载器

负责加载和解析API文档。

重构前: APIDocumentationSearch中的加载逻辑 (~30行)
重构后: DocumentLoader独立组件 (~25行)
"""

import json
from typing import Dict, Any


class DocumentLoader:
    """
    文档加载器
    
    职责：
    - 加载API文档文件
    - 解析文档结构
    - 验证文档格式
    """
    
    def load_documents(self, docs_file: str) -> Dict[str, Dict[str, Any]]:
        """
        加载API文档

        Args:
            docs_file: 文档文件路径

        Returns:
            Dict[str, Dict[str, Any]]: 文档字典
        """
        return self.load_from_file(docs_file)

    def load_from_file(self, docs_file: str) -> Dict[str, Dict[str, Any]]:
        """
        加载API文档
        
        Args:
            docs_file: 文档文件路径
        
        Returns:
            Dict[str, Dict[str, Any]]: 文档字典
        """
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        documents = {}
        
        # 解析文档结构
        if 'endpoints' in docs_data:
            for endpoint_key, endpoint_data in docs_data['endpoints'].items():
                documents[endpoint_key] = endpoint_data
        
        print(f"✅ 已加载 {len(documents)} 个API端点文档")
        
        return documents

