"""
api_documentation_search 模块

提供 api_documentation_search 相关功能和接口。
"""

import json
import os
import re

import heapq

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API文档搜索和导航优化系统
提供高效的API文档检索和导航功能
"""


@dataclass
class SearchResult:
    """搜索结果"""
    endpoint_id: str
    title: str
    description: str
    relevance_score: float
    match_type: str  # exact, partial, fuzzy
    matched_fields: List[str]
    snippet: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = ""


@dataclass
class NavigationIndex:
    """导航索引"""
    categories: Dict[str, List[str]] = field(default_factory=dict)
    tags: Dict[str, List[str]] = field(default_factory=dict)
    endpoints_by_method: Dict[str, List[str]] = field(default_factory=dict)
    endpoints_by_status: Dict[str, List[str]] = field(default_factory=dict)
    parameter_index: Dict[str, List[str]] = field(default_factory=dict)
    response_code_index: Dict[str, List[str]] = field(default_factory=dict)


class APIDocumentationSearch:
    """API文档搜索器"""

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.index: NavigationIndex = NavigationIndex()
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self._stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'api', 'get', 'post', 'put', 'delete', 'http'
        }

    def load_documents(self, docs_file: str):
        """加载API文档"""
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        self.documents = {}

        # 解析文档结构
        if 'endpoints' in docs_data:
            for endpoint_key, endpoint_data in docs_data['endpoints'].items():
                self.documents[endpoint_key] = endpoint_data

        # 构建导航索引
        self._build_navigation_index()

        print(f"已加载 {len(self.documents)} 个API端点文档")

    def search(self, query: str, limit: int = 20,
               search_type: str = "all") -> List[SearchResult]:
        """搜索API文档"""
        if not query.strip():
            return []

        # 检查缓存
        cache_key = f"{query}_{limit}_{search_type}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        # 执行搜索
        results = []

        # 分词查询
        query_terms = self._tokenize_query(query.lower())

        for endpoint_id, doc_data in self.documents.items():
            score = self._calculate_relevance_score(query_terms, doc_data, search_type)
            if score > 0:
                result = SearchResult(
                    endpoint_id=endpoint_id,
                    title=doc_data.get('summary', ''),
                    description=doc_data.get('description', ''),
                    relevance_score=score,
                    match_type=self._determine_match_type(query_terms, doc_data),
                    matched_fields=self._find_matched_fields(query_terms, doc_data),
                    snippet=self._generate_snippet(query_terms, doc_data),
                    tags=self._extract_tags(doc_data),
                    category=self._categorize_endpoint(endpoint_id)
                )
                results.append(result)

        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # 限制结果数量
        results = results[:limit]

        # 缓存结果
        self.search_cache[cache_key] = results

        return results

    def _tokenize_query(self, query: str) -> List[str]:
        """分词查询"""
        # 移除标点符号
        query = re.sub(r'[^\w\s]', ' ', query)

        # 分词
        tokens = query.split()

        # 移除停用词
        tokens = [token for token in tokens if token not in self._stop_words]

        return tokens

    def _calculate_relevance_score(self, query_terms: List[str],
                                   doc_data: Dict[str, Any],
                                   search_type: str) -> float:
        """计算相关性分数"""
        score = 0.0

        # 根据搜索类型计算不同部分的得分
        if self._should_search_endpoints(search_type):
            score += self._calculate_endpoint_score(query_terms, doc_data)

        if self._should_search_parameters(search_type):
            score += self._calculate_parameters_score(query_terms, doc_data)

        if self._should_search_responses(search_type):
            score += self._calculate_responses_score(query_terms, doc_data)

        return score

    def _should_search_endpoints(self, search_type: str) -> bool:
        """判断是否应该搜索端点信息"""
        return search_type in ["all", "endpoint"]

    def _should_search_parameters(self, search_type: str) -> bool:
        """判断是否应该搜索参数信息"""
        return search_type in ["all", "parameters"]

    def _should_search_responses(self, search_type: str) -> bool:
        """判断是否应该搜索响应信息"""
        return search_type in ["all", "responses"]

    def _calculate_endpoint_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算端点相关性得分"""
        score = 0.0
        text_fields = ['summary', 'description']

        for field in text_fields:
            if field in doc_data:
                field_text = doc_data[field].lower()
                field_score = self._calculate_field_score(query_terms, field_text, field)
                score += field_score

        return score

    def _calculate_field_score(self, query_terms: List[str], field_text: str, field_name: str) -> float:
        """计算字段匹配得分"""
        score = 0.0
        weight = 2.0 if field_name == 'summary' else 1.0

        for term in query_terms:
            if term in field_text:
                score += weight

        return score

    def _calculate_parameters_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算参数相关性得分"""
        score = 0.0

        if 'parameters' not in doc_data:
            return score

        for param in doc_data['parameters']:
            param_text = json.dumps(param, ensure_ascii=False).lower()
            param_score = self._calculate_text_score(query_terms, param_text, 1.5)
            score += param_score

        return score

    def _calculate_responses_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算响应相关性得分"""
        score = 0.0

        if 'responses' not in doc_data:
            return score

        for response in doc_data['responses']:
            response_text = json.dumps(response, ensure_ascii=False).lower()
            response_score = self._calculate_text_score(query_terms, response_text, 1.0)
            score += response_score

        return score

    def _calculate_text_score(self, query_terms: List[str], text: str, weight: float) -> float:
        """计算文本匹配得分"""
        score = 0.0
        for term in query_terms:
            if term in text:
                score += weight
        return score

    def _determine_match_type(self, query_terms: List[str],
                              doc_data: Dict[str, Any]) -> str:
        """确定匹配类型"""
        # 检查精确匹配
        for term in query_terms:
            if (term in doc_data.get('summary', '').lower() or
                    term in doc_data.get('description', '').lower()):
                return "exact"

        # 检查部分匹配
        for term in query_terms:
            if (term in doc_data.get('summary', '').lower() or
                    term in doc_data.get('description', '').lower()):
                return "partial"

        return "fuzzy"

    def _find_matched_fields(self, query_terms: List[str],
                             doc_data: Dict[str, Any]) -> List[str]:
        """查找匹配的字段"""
        matched_fields = []

        for term in query_terms:
            if term in doc_data.get('summary', '').lower():
                matched_fields.append('summary')
            if term in doc_data.get('description', '').lower():
                matched_fields.append('description')
            if 'parameters' in doc_data:
                for param in doc_data['parameters']:
                    if term in json.dumps(param, ensure_ascii=False).lower():
                        matched_fields.append('parameters')
            if 'responses' in doc_data:
                for response in doc_data['responses']:
                    if term in json.dumps(response, ensure_ascii=False).lower():
                        matched_fields.append('responses')

        return list(set(matched_fields))

    def _generate_snippet(self, query_terms: List[str],
                          doc_data: Dict[str, Any]) -> str:
        """生成搜索片段"""
        text = doc_data.get('description', doc_data.get('summary', ''))

        # 查找包含查询词的句子
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            if any(term in sentence.lower() for term in query_terms):
                # 高亮查询词
                for term in query_terms:
                    sentence = re.sub(
                        f'({re.escape(term)})',
                        r'**\1**',
                        sentence,
                        flags=re.IGNORECASE
                    )
                return sentence.strip()

        # 如果没有找到匹配的句子，返回前100个字符
        return text[:100] + "..." if len(text) > 100 else text

    def _extract_tags(self, doc_data: Dict[str, Any]) -> List[str]:
        """提取标签"""
        tags = []

        # 从方法和路径提取标签
        if 'method' in doc_data:
            tags.append(doc_data['method'].lower())

        # 从错误码提取标签
        if 'error_codes' in doc_data:
            for error in doc_data['error_codes']:
                if 'category' in error:
                    tags.append(error['category'].lower())

        return tags

    def _categorize_endpoint(self, endpoint_id: str) -> str:
        """分类端点"""
        if 'data' in endpoint_id.lower():
            return "数据服务"
        elif 'trading' in endpoint_id.lower():
            return "交易服务"
        elif 'features' in endpoint_id.lower():
            return "特征工程"
        elif 'monitoring' in endpoint_id.lower():
            return "监控服务"
        else:
            return "其他"

    def _build_navigation_index(self):
        """构建导航索引"""
        self.index = NavigationIndex()

        for endpoint_id, doc_data in self.documents.items():
            # 按方法索引
            method = doc_data.get('method', 'GET')
            if method not in self.index.endpoints_by_method:
                self.index.endpoints_by_method[method] = []
            self.index.endpoints_by_method[method].append(endpoint_id)

            # 按状态码索引
            if 'responses' in doc_data:
                for response in doc_data['responses']:
                    status_code = str(response.get('status_code', ''))
                    if status_code not in self.index.endpoints_by_status:
                        self.index.endpoints_by_status[status_code] = []
                    self.index.endpoints_by_status[status_code].append(endpoint_id)

            # 参数索引
            if 'parameters' in doc_data:
                for param in doc_data['parameters']:
                    param_name = param.get('name', '')
                    if param_name not in self.index.parameter_index:
                        self.index.parameter_index[param_name] = []
                    self.index.parameter_index[param_name].append(endpoint_id)

    def get_navigation_suggestions(self, current_path: str = "") -> Dict[str, Any]:
        """获取导航建议"""
        suggestions = {
            "popular_endpoints": [],
            "related_endpoints": [],
            "method_distribution": {},
            "category_distribution": {}
        }

        # 热门端点（基于响应数量）
        endpoint_popularity = []
        for endpoint_id, doc_data in self.documents.items():
            popularity_score = len(doc_data.get('responses', []))
            heapq.heappush(endpoint_popularity, (popularity_score, endpoint_id))

        suggestions["popular_endpoints"] = [
            endpoint_id for _, endpoint_id in heapq.nlargest(5, endpoint_popularity)
        ]

        # 相关端点（基于当前路径）
        if current_path:
            path_parts = current_path.lower().split('/')
            related = []
            for endpoint_id in self.documents.keys():
                if any(part in endpoint_id.lower() for part in path_parts if part):
                    related.append(endpoint_id)
            suggestions["related_endpoints"] = related[:5]

        # 方法分布
        suggestions["method_distribution"] = self.index.endpoints_by_method.copy()

        # 类别分布
        category_count = defaultdict(int)
        for endpoint_id in self.documents.keys():
            category = self._categorize_endpoint(endpoint_id)
            category_count[category] += 1
        suggestions["category_distribution"] = dict(category_count)

        return suggestions

    def get_endpoint_details(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """获取端点详细信息"""
        return self.documents.get(endpoint_id)

    def get_endpoints_by_category(self, category: str) -> List[str]:
        """按类别获取端点"""
        return [
            endpoint_id for endpoint_id in self.documents.keys()
            if self._categorize_endpoint(endpoint_id) == category
        ]

    def get_endpoints_by_method(self, method: str) -> List[str]:
        """按方法获取端点"""
        return self.index.endpoints_by_method.get(method.upper(), [])

    def get_endpoints_by_parameter(self, parameter: str) -> List[str]:
        """按参数获取端点"""
        return self.index.parameter_index.get(parameter, [])

    def clear_cache(self):
        """清除搜索缓存"""
        self.search_cache.clear()

    def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            "total_documents": len(self.documents),
            "cached_queries": len(self.search_cache),
            "methods_count": len(self.index.endpoints_by_method),
            "parameters_count": len(self.index.parameter_index),
            "categories_count": len(set(
                self._categorize_endpoint(eid) for eid in self.documents.keys()
            ))
        }


class APIDocumentationBrowser:
    """API文档浏览器"""

    def __init__(self, search_engine: APIDocumentationSearch):
        self.search_engine = search_engine
        self.browse_history: List[str] = []
        self.bookmarks: Set[str] = set()

    def browse_endpoint(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """浏览端点"""
        details = self.search_engine.get_endpoint_details(endpoint_id)
        if details:
            self.browse_history.append(endpoint_id)
            # 保持最近10个浏览记录
            if len(self.browse_history) > 10:
                self.browse_history = self.browse_history[-10:]
        return details

    def add_bookmark(self, endpoint_id: str):
        """添加书签"""
        self.bookmarks.add(endpoint_id)

    def remove_bookmark(self, endpoint_id: str):
        """移除书签"""
        self.bookmarks.discard(endpoint_id)

    def get_bookmarks(self) -> List[str]:
        """获取书签列表"""
        return list(self.bookmarks)

    def get_browse_history(self) -> List[str]:
        """获取浏览历史"""
        return self.browse_history.copy()

    def get_recently_viewed(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近查看的端点"""
        recent_endpoints = []
        for endpoint_id in reversed(self.browse_history[-limit:]):
            details = self.search_engine.get_endpoint_details(endpoint_id)
            if details:
                recent_endpoints.append({
                    "id": endpoint_id,
                    "summary": details.get("summary", ""),
                    "method": details.get("method", "GET")
                })
        return recent_endpoints


def create_api_documentation_search(docs_file: str) -> APIDocumentationSearch:
    """创建API文档搜索实例"""
    search_engine = APIDocumentationSearch()
    search_engine.load_documents(docs_file)
    return search_engine


if __name__ == "__main__":
    # 创建API文档搜索系统
    print("初始化API文档搜索系统...")

    # 假设文档文件路径
    docs_file = "docs/api/enhanced_rqa_api_documentation.json"

    if os.path.exists(docs_file):
        search_engine = create_api_documentation_search(docs_file)

        # 创建浏览器
        browser = APIDocumentationBrowser(search_engine)

        # 执行搜索
        print("执行示例搜索...")

        # 搜索数据相关的API
        results = search_engine.search("market data", limit=5)
        print(f"\\n搜索 'market data' 结果 ({len(results)} 个):")
        for result in results[:3]:
            print(f"  • {result.title} (相关性: {result.relevance_score:.2f})")

        # 搜索交易相关的API
        results = search_engine.search("strategy", limit=5)
        print(f"\\n搜索 'strategy' 结果 ({len(results)} 个):")
        for result in results[:3]:
            print(f"  • {result.title} (相关性: {result.relevance_score:.2f})")

        # 获取导航建议
        suggestions = search_engine.get_navigation_suggestions()
        print(f"\\n导航建议:")
        print(f"  📈 热门端点: {len(suggestions['popular_endpoints'])} 个")
        print(f"  🔗 方法分布: {suggestions['method_distribution']}")
        print(f"  📊 类别分布: {suggestions['category_distribution']}")

        # 获取搜索统计
        stats = search_engine.get_search_statistics()
        print(f"\\n搜索统计: {stats}")

        print("\\n🎉 API文档搜索系统测试完成！")
    else:
        print(f"文档文件不存在: {docs_file}")
        print("请先运行API文档生成器生成文档文件")
