"""
导航索引构建器

负责构建API文档的导航索引，支持快速查询和过滤。

重构前: APIDocumentationSearch中的索引逻辑 (~100行)
重构后: NavigationBuilder独立组件 (~80行)
"""

from typing import Dict, Any, List
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class NavigationIndex:
    """导航索引"""
    categories: Dict[str, List[str]] = field(default_factory=dict)
    tags: Dict[str, List[str]] = field(default_factory=dict)
    endpoints_by_method: Dict[str, List[str]] = field(default_factory=dict)
    endpoints_by_status: Dict[str, List[str]] = field(default_factory=dict)
    parameter_index: Dict[str, List[str]] = field(default_factory=dict)
    response_code_index: Dict[str, List[str]] = field(default_factory=dict)


class NavigationBuilder:
    """
    导航索引构建器
    
    职责：
    - 构建分类索引
    - 构建标签索引
    - 构建方法索引
    - 构建参数索引
    """
    
    def build_navigation_index(self, documents: Dict[str, Dict[str, Any]]) -> NavigationIndex:
        """
        构建导航索引
        
        Args:
            documents: 文档集合
        
        Returns:
            NavigationIndex: 导航索引对象
        """
        index = NavigationIndex()
        
        # 构建各类索引
        index.categories = self._build_category_index(documents)
        index.tags = self._build_tag_index(documents)
        index.endpoints_by_method = self._build_method_index(documents)
        index.parameter_index = self._build_parameter_index(documents)
        index.response_code_index = self._build_response_index(documents)
        
        return index
    
    def _build_category_index(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建分类索引"""
        categories = defaultdict(list)
        
        for doc_id, doc_data in documents.items():
            category = self._categorize_endpoint(doc_id, doc_data)
            categories[category].append(doc_id)
        
        return dict(categories)
    
    def _build_tag_index(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建标签索引"""
        tags = defaultdict(list)
        
        for doc_id, doc_data in documents.items():
            for tag in doc_data.get('tags', []):
                tags[tag].append(doc_id)
        
        return dict(tags)
    
    def _build_method_index(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建方法索引"""
        methods = defaultdict(list)
        
        for doc_id, doc_data in documents.items():
            method = doc_data.get('method', 'GET').upper()
            methods[method].append(doc_id)
        
        return dict(methods)
    
    def _build_parameter_index(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建参数索引"""
        params = defaultdict(list)
        
        for doc_id, doc_data in documents.items():
            for param in doc_data.get('parameters', []):
                param_name = param.get('name', '')
                if param_name:
                    params[param_name].append(doc_id)
        
        return dict(params)
    
    def _build_response_index(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建响应码索引"""
        responses = defaultdict(list)
        
        for doc_id, doc_data in documents.items():
            for code in doc_data.get('responses', {}).keys():
                responses[str(code)].append(doc_id)
        
        return dict(responses)

    def build_navigation(self, current_path: str = "") -> Dict[str, Any]:
        """
        构建导航建议

        Args:
            current_path: 当前路径

        Returns:
            Dict[str, Any]: 导航建议
        """
        if not hasattr(self, '_documents') or not self._documents:
            return {"categories": [], "methods": [], "tags": []}

        # 构建导航索引
        nav_index = self.build_navigation_index(self._documents)

        return {
            "categories": list(nav_index.categories.keys()) if hasattr(nav_index, 'categories') else [],
            "methods": ["GET", "POST", "PUT", "DELETE"],  # 默认HTTP方法
            "tags": list(nav_index.tags.keys()) if hasattr(nav_index, 'tags') else [],
            "current_path": current_path,
            "total_endpoints": len(self._documents)
        }

    def clear_cache(self):
        """清除缓存"""
        self._documents = {}
        self._navigation_index = None

    def rebuild_index(self):
        """重建索引"""
        if hasattr(self, '_documents') and self._documents:
            self._navigation_index = self.build_navigation_index(self._documents)
    
    def _categorize_endpoint(self, endpoint_id: str, doc_data: Dict[str, Any]) -> str:
        """分类端点"""
        explicit_category = doc_data.get('category')
        if explicit_category:
            return explicit_category

        path = doc_data.get('path', '').lower()
        
        if '/data/' in path:
            return 'data_service'
        elif '/feature' in path:
            return 'feature_service'
        elif '/trading/' in path or '/order' in path:
            return 'trading_service'
        elif '/health' in path or '/metric' in path:
            return 'monitoring_service'
        else:
            return 'other'

