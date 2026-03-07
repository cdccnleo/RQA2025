"""
搜索引擎核心

负责API文档的全文搜索和相关性评分。

重构前: APIDocumentationSearch中的搜索逻辑 (~200行)
重构后: SearchEngine独立组件 (~150行)
"""

import re
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field


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


class SearchEngine:
    """
    搜索引擎核心
    
    职责：
    - 全文搜索
    - 相关性评分
    - 结果排序和过滤
    """
    
    def __init__(self):
        """初始化搜索引擎"""
        self._stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'api', 'get', 'post', 'put', 'delete', 'http'
        }
        self.search_cache: Dict[str, List[SearchResult]] = {}
    
    def search(
        self,
        query: str,
        documents: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 20,
        search_type: str = "all"
    ) -> List[SearchResult]:
        """
        执行搜索

        Args:
            query: 搜索查询
            documents: 文档集合
            limit: 结果限制
            search_type: 搜索类型 (all, endpoints, parameters, responses)

        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 检查缓存
        documents = documents or {}
        cache_key = f"{query}_{search_type}_{limit}"
        cached_result = self._check_search_cache(cache_key)
        self._search_count = getattr(self, "_search_count", 0) + 1
        if cached_result is not None:
            self._cache_hits = getattr(self, "_cache_hits", 0) + 1
            self._cache_hit_rate = self._cache_hits / max(self._search_count, 1)
            return cached_result

        # 预处理查询
        query_terms = self._preprocess_query(query)
        if not query_terms:
            return []

        # 执行搜索
        results = self._execute_search(query_terms, documents, search_type)

        # 处理结果
        final_results = self._process_search_results(results, limit)

        # 缓存结果
        self._cache_search_results(cache_key, final_results)
        self._cache_hit_rate = getattr(self, "_cache_hits", 0) / max(self._search_count, 1)

        return final_results

    def _check_search_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """检查搜索缓存"""
        return self.search_cache.get(cache_key)

    def _preprocess_query(self, query: str) -> List[str]:
        """预处理查询"""
        query_terms = self._tokenize_query(query)
        return query_terms if query_terms else []

    def _execute_search(self, query_terms: List[str], documents: Dict[str, Dict[str, Any]],
                       search_type: str) -> List[SearchResult]:
        """执行搜索操作"""
        results = []
        for doc_id, doc_data in documents.items():
            score = self._calculate_relevance_score(query_terms, doc_data, search_type)

            if score > 0:
                result = self._create_search_result(doc_id, doc_data, query_terms, score)
                results.append(result)

        return results

    def _process_search_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        """处理搜索结果"""
        # 排序
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # 限制结果数量
        return results[:limit]

    def _cache_search_results(self, cache_key: str, results: List[SearchResult]):
        """缓存搜索结果"""
        self.search_cache[cache_key] = results
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        分词查询
        
        原方法: _tokenize_query (12行)
        新方法: 保持不变 (~12行)
        """
        # 转小写
        query = query.lower()
        
        # 分词
        terms = re.findall(r'\w+', query)
        
        # 移除停用词
        terms = [t for t in terms if t not in self._stop_words and len(t) > 1]
        
        return terms
    
    def _calculate_relevance_score(
        self,
        query_terms: List[str],
        doc_data: Dict[str, Any],
        search_type: str
    ) -> float:
        """
        计算相关性评分
        
        原方法: _calculate_relevance_score (17行)
        新方法: 协调器方法 (~20行)
        """
        score = 0.0
        
        # 根据搜索类型选择评分方法
        if self._should_search_endpoints(search_type):
            score += self._calculate_endpoint_score(query_terms, doc_data)
        
        if self._should_search_parameters(search_type):
            score += self._calculate_parameters_score(query_terms, doc_data)
        
        if self._should_search_responses(search_type):
            score += self._calculate_responses_score(query_terms, doc_data)
        
        return score
    
    def _should_search_endpoints(self, search_type: str) -> bool:
        """是否搜索端点"""
        return search_type in ("all", "endpoints")
    
    def _should_search_parameters(self, search_type: str) -> bool:
        """是否搜索参数"""
        return search_type in ("all", "parameters")
    
    def _should_search_responses(self, search_type: str) -> bool:
        """是否搜索响应"""
        return search_type in ("all", "responses")
    
    def _calculate_endpoint_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算端点评分"""
        score = 0.0
        
        # 路径匹配（权重3.0）
        score += self._calculate_field_score(query_terms, doc_data.get('path', ''), 'path') * 3.0
        
        # 摘要匹配（权重2.0）
        score += self._calculate_field_score(query_terms, doc_data.get('summary', ''), 'summary') * 2.0
        
        # 描述匹配（权重1.5）
        score += self._calculate_field_score(query_terms, doc_data.get('description', ''), 'description') * 1.5
        
        return score
    
    def _calculate_field_score(self, query_terms: List[str], field_text: str, field_name: str) -> float:
        """计算字段评分"""
        if not field_text:
            return 0.0
        
        field_lower = field_text.lower()
        score = 0.0
        
        for term in query_terms:
            if term in field_lower:
                score += 1.0
        
        return score
    
    def _calculate_parameters_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算参数评分"""
        score = 0.0
        parameters = doc_data.get('parameters', [])
        
        for param in parameters:
            param_text = f"{param.get('name', '')} {param.get('description', '')}"
            score += self._calculate_text_score(query_terms, param_text, 1.0)
        
        return score
    
    def _calculate_responses_score(self, query_terms: List[str], doc_data: Dict[str, Any]) -> float:
        """计算响应评分"""
        score = 0.0
        responses = doc_data.get('responses', {})
        
        for resp_data in responses.values():
            resp_text = resp_data.get('description', '')
            score += self._calculate_text_score(query_terms, resp_text, 0.5)
        
        return score
    
    def _calculate_text_score(self, query_terms: List[str], text: str, weight: float) -> float:
        """计算文本评分"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        
        return matches * weight
    
    def _create_search_result(
        self,
        doc_id: str,
        doc_data: Dict[str, Any],
        query_terms: List[str],
        score: float
    ) -> SearchResult:
        """创建搜索结果对象"""
        match_type = self._determine_match_type(query_terms, doc_data)
        matched_fields = self._find_matched_fields(query_terms, doc_data)
        snippet = self._generate_snippet(query_terms, doc_data)
        
        return SearchResult(
            endpoint_id=doc_id,
            title=doc_data.get('summary', ''),
            description=doc_data.get('description', ''),
            relevance_score=score,
            match_type=match_type,
            matched_fields=matched_fields,
            snippet=snippet,
            tags=self._extract_tags(doc_data),
            category=self._categorize_endpoint(doc_id)
        )
    
    def _determine_match_type(self, query_terms: List[str], doc_data: Dict[str, Any]) -> str:
        """确定匹配类型"""
        text = f"{doc_data.get('path', '')} {doc_data.get('summary', '')}".lower()
        
        # 精确匹配
        for term in query_terms:
            if term in text:
                return 'exact'
        
        # 部分匹配
        return 'partial'
    
    def _find_matched_fields(self, query_terms: List[str], doc_data: Dict[str, Any]) -> List[str]:
        """查找匹配的字段"""
        matched = []
        
        for field in ['path', 'summary', 'description']:
            field_text = doc_data.get(field, '').lower()
            if any(term in field_text for term in query_terms):
                matched.append(field)
        
        return matched
    
    def _generate_snippet(self, query_terms: List[str], doc_data: Dict[str, Any]) -> str:
        """生成摘要片段"""
        description = doc_data.get('description', '')
        if not description:
            return doc_data.get('summary', '')[:100]
        
        # 简单截取
        return description[:200] + "..." if len(description) > 200 else description
    
    def _extract_tags(self, doc_data: Dict[str, Any]) -> List[str]:
        """提取标签"""
        return doc_data.get('tags', [])
    
    def _categorize_endpoint(self, endpoint_id: str) -> str:
        """分类端点"""
        if 'data' in endpoint_id:
            return 'data_service'
        elif 'feature' in endpoint_id:
            return 'feature_service'
        elif 'trading' in endpoint_id or 'order' in endpoint_id:
            return 'trading_service'
        elif 'monitor' in endpoint_id or 'health' in endpoint_id:
            return 'monitoring_service'
        else:
            return 'other'
    
    def clear_cache(self):
        """清空搜索缓存"""
        self.search_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            "total_searches": getattr(self, '_search_count', 0),
            "cache_hit_rate": getattr(self, '_cache_hit_rate', 0.0),
            "avg_search_time": getattr(self, '_avg_search_time', 0.0),
            "total_cached_results": len(self.search_cache),
            "supported_search_types": ["all", "endpoints", "parameters", "responses"]
        }

