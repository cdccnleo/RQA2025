"""
OpenAPI文档组装器

负责将各个组件构建的内容组装成完整的OpenAPI文档
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import yaml


class APISchema:
    """API Schema临时类（保持兼容）"""
    
    def __init__(self, title: str, version: str, description: str,
                 servers: List[Dict] = None, security_schemes: Dict = None):
        self.title = title
        self.version = version
        self.description = description
        self.servers = servers or []
        self.security_schemes = security_schemes or {}
        self.endpoints = []
        self.schemas = {}


class DocumentationAssembler:
    """
    文档组装器
    
    职责：
    - 组装API基本信息
    - 整合所有端点定义
    - 整合所有Schema定义
    - 生成最终OpenAPI规范
    """
    
    def __init__(self):
        """初始化文档组装器"""
        self.api_info = {}
        self.endpoints = []
        self.schemas = {}
        self.security_schemes = {}
        self.servers = []
        self.tags = []
    
    def set_api_info(
        self,
        title: str,
        version: str,
        description: str
    ):
        """设置API基本信息"""
        self.api_info = {
            'title': title,
            'version': version,
            'description': description
        }
    
    def add_servers(self, servers: List[Dict[str, str]]):
        """添加服务器配置"""
        self.servers.extend(servers)
    
    def add_security_schemes(self, schemes: Dict[str, Dict]):
        """添加安全方案"""
        self.security_schemes.update(schemes)
    
    def add_endpoints(self, endpoints: List[Any]):
        """添加端点列表"""
        self.endpoints.extend(endpoints)
    
    def add_schemas(self, schemas: Dict[str, Dict]):
        """添加Schema定义"""
        self.schemas.update(schemas)
    
    def add_tags(self, tags: List[Dict[str, str]]):
        """添加标签定义"""
        self.tags.extend(tags)
    
    def assemble(self) -> Dict[str, Any]:
        """
        组装完整的OpenAPI文档
        
        Returns:
            Dict[str, Any]: OpenAPI 3.0规范文档
        """
        openapi_doc = {
            'openapi': '3.0.0',
            'info': self.api_info,
            'servers': self.servers,
            'paths': self._assemble_paths(),
            'components': {
                'schemas': self.schemas,
                'securitySchemes': self.security_schemes
            },
            'tags': self.tags,
            'security': [{'bearerAuth': []}] if 'bearerAuth' in self.security_schemes else []
        }
        
        return openapi_doc
    
    def _assemble_paths(self) -> Dict[str, Dict]:
        """组装路径定义"""
        paths = {}
        
        for endpoint in self.endpoints:
            path = endpoint.path
            method = endpoint.method.lower()
            
            if path not in paths:
                paths[path] = {}
            
            paths[path][method] = {
                'summary': endpoint.summary,
                'description': endpoint.description,
                'tags': endpoint.tags,
                'parameters': endpoint.parameters,
                'responses': endpoint.responses
            }
            
            if endpoint.request_body:
                paths[path][method]['requestBody'] = endpoint.request_body
        
        return paths
    
    def export_to_json(self, file_path: str, pretty: bool = True):
        """导出为JSON文件"""
        doc = self.assemble()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2 if pretty else None, ensure_ascii=False)
        
        print(f"✅ OpenAPI文档已导出到: {file_path}")
    
    def export_to_yaml(self, file_path: str):
        """导出为YAML文件"""
        doc = self.assemble()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(doc, f, allow_unicode=True, sort_keys=False)
        
        print(f"✅ OpenAPI文档已导出到: {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        return {
            'total_endpoints': len(self.endpoints),
            'total_schemas': len(self.schemas),
            'total_servers': len(self.servers),
            'total_tags': len(self.tags),
            'endpoints_by_method': self._count_by_method(),
            'endpoints_by_tag': self._count_by_tag()
        }
    
    def _count_by_method(self) -> Dict[str, int]:
        """按HTTP方法统计端点"""
        counts = {}
        for endpoint in self.endpoints:
            method = endpoint.method.upper()
            counts[method] = counts.get(method, 0) + 1
        return counts
    
    def _count_by_tag(self) -> Dict[str, int]:
        """按标签统计端点"""
        counts = {}
        for endpoint in self.endpoints:
            for tag in endpoint.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

