"""
优化引擎组件

负责分析监控数据并生成优化建议。
"""

from datetime import datetime
from typing import Dict, Any, List, Optional


class OptimizationEngine:
    """优化引擎"""
    
    def __init__(self):
        """初始化优化引擎"""
        self.optimization_suggestions: List[Dict[str, Any]] = []
    
    def generate_suggestions(self, coverage_data: Dict[str, Any], 
                           performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        print("💡 生成优化建议...")

        suggestions = []

        # 基于覆盖率数据生成建议
        coverage_suggestions = self._generate_coverage_suggestions(coverage_data)
        suggestions.extend(coverage_suggestions)

        # 基于性能数据生成建议
        performance_suggestions = self._generate_performance_suggestions(performance_data)
        suggestions.extend(performance_suggestions)

        # 保存建议
        self.optimization_suggestions.extend(suggestions)

        # 打印建议
        self._print_suggestions(suggestions)

        return suggestions
    
    def _generate_coverage_suggestions(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于覆盖率数据生成建议"""
        suggestions = []
        
        try:
            coverage_percent = coverage_data.get('coverage_percent', 0)
            
            if coverage_percent < 80:
                suggestions.append({
                    'type': 'coverage_improvement',
                    'priority': 'high',
                    'title': '提升测试覆盖率',
                    'description': f'当前覆盖率仅为{coverage_percent:.1f}%，建议补充单元测试',
                    'actions': [
                        '为src/engine/和src/features/目录添加单元测试',
                        '实现ML模型的单元测试',
                        '完善集成测试用例'
                    ],
                    'timestamp': datetime.now()
                })
            
            elif coverage_percent < 90:
                suggestions.append({
                    'type': 'coverage_improvement',
                    'priority': 'medium',
                    'title': '进一步提升测试覆盖率',
                    'description': f'当前覆盖率{coverage_percent:.1f}%，建议提升到90%以上',
                    'actions': [
                        '补充边界条件测试',
                        '添加异常处理测试',
                        '完善端到端测试'
                    ],
                    'timestamp': datetime.now()
                })
        except (KeyError, TypeError) as e:
            print(f"❌ 生成覆盖率建议时发生错误: {e}")
        
        return suggestions
    
    def _generate_performance_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于性能数据生成建议"""
        suggestions = []
        
        try:
            # 生成各类性能建议
            suggestions.extend(self._check_response_time_suggestions(performance_data))
            suggestions.extend(self._check_memory_usage_suggestions(performance_data))
            suggestions.extend(self._check_throughput_suggestions(performance_data))
        except (KeyError, TypeError, ZeroDivisionError) as e:
            print(f"❌ 生成性能建议时发生错误: {e}")
        
        return suggestions
    
    def _check_response_time_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查响应时间建议"""
        suggestions = []
        
        response_time = performance_data.get('response_time_ms', 0)
        if response_time > 10:
            suggestions.append({
                'type': 'performance_optimization',
                'priority': 'medium',
                'title': '优化响应时间',
                'description': f'当前响应时间{response_time:.1f}ms，建议优化',
                'actions': [
                    '优化缓存策略',
                    '改进数据库查询',
                    '启用异步处理',
                    '实现负载均衡'
                ],
                'timestamp': datetime.now()
            })
        
        return suggestions
    
    def _check_memory_usage_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查内存使用建议"""
        suggestions = []
        
        memory_usage_mb = performance_data.get('memory_usage_mb', 0)
        memory_usage_gb = memory_usage_mb / 1024
        if memory_usage_gb > 1:
            suggestions.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'title': '优化内存使用',
                'description': f'内存使用{memory_usage_gb:.1f}GB，建议优化',
                'actions': [
                    '实现内存池化',
                    '优化对象生命周期',
                    '启用垃圾回收调优',
                    '检查内存泄漏'
                ],
                'timestamp': datetime.now()
            })
        
        return suggestions
    
    def _check_throughput_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查吞吐量建议"""
        suggestions = []
        
        throughput = performance_data.get('throughput_tps', 0)
        if throughput < 1000:
            suggestions.append({
                'type': 'throughput_optimization',
                'priority': 'medium',
                'title': '提升系统吞吐量',
                'description': f'当前吞吐量{throughput} tps，建议优化',
                'actions': [
                    '优化数据库连接池',
                    '实现并发处理',
                    '使用更高效的序列化',
                    '优化网络IO'
                ],
                'timestamp': datetime.now()
            })
        
        return suggestions
    
    def _print_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """打印优化建议"""
        for suggestion in suggestions:
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            emoji = priority_emoji.get(suggestion.get('priority', ''), '⚪')
            title = suggestion.get('title', '未知建议')
            priority = suggestion.get('priority', 'unknown')
            print(f"{emoji} {priority.upper()}: {title}")
    
    def get_suggestions_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取建议历史"""
        if limit is None:
            return self.optimization_suggestions.copy()
        return self.optimization_suggestions[-limit:] if limit > 0 else []
    
    def clear_suggestions_history(self) -> None:
        """清空建议历史"""
        self.optimization_suggestions.clear()
    
    def get_latest_suggestions(self, count: int = 3) -> List[Dict[str, Any]]:
        """获取最新的优化建议"""
        return self.optimization_suggestions[-count:] if count > 0 else []
