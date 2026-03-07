"""
AI质量保障连续学习与优化系统

实现AI系统的持续学习、模型优化和知识积累：
1. 学习循环管理 - 基于反馈的持续学习机制
2. 模型自适应优化 - 根据新数据自动调整模型
3. 知识库积累 - 质量保障知识的持续积累和更新
4. 性能趋势学习 - 学习历史模式预测未来趋势
5. 异常模式进化 - 基于新异常的模式学习和更新
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import asyncio
import threading
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LearningLoopManager:
    """学习循环管理器"""

    def __init__(self, learning_config: Dict[str, Any] = None):
        self.config = learning_config or self._get_default_config()
        self.learning_cycles = []
        self.feedback_buffer = []
        self.learning_active = False
        self.learning_thread = None

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'learning_interval': 3600,  # 1小时学习一次
            'feedback_buffer_size': 1000,
            'min_samples_for_learning': 100,
            'learning_timeout': 1800,  # 30分钟学习超时
            'performance_improvement_threshold': 0.05,  # 5%性能提升阈值
            'auto_learning_enabled': True,
            'manual_review_required': False
        }

    def start_learning_loop(self):
        """启动学习循环"""
        if self.learning_active:
            return

        self.learning_active = True

        def learning_worker():
            while self.learning_active:
                try:
                    # 执行学习周期
                    self._execute_learning_cycle()

                    # 等待下一个学习周期
                    time.sleep(self.config['learning_interval'])

                except Exception as e:
                    logger.error(f"学习循环执行失败: {e}")
                    time.sleep(self.config['learning_interval'])

        self.learning_thread = threading.Thread(target=learning_worker, daemon=True)
        self.learning_thread.start()

        logger.info("学习循环已启动")

    def stop_learning_loop(self):
        """停止学习循环"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10)
        logger.info("学习循环已停止")

    def add_feedback(self, feedback_data: Dict[str, Any]):
        """添加反馈数据"""
        try:
            feedback_entry = {
                'timestamp': datetime.now(),
                'feedback_type': feedback_data.get('type', 'general'),
                'data': feedback_data,
                'processed': False
            }

            self.feedback_buffer.append(feedback_entry)

            # 限制缓冲区大小
            if len(self.feedback_buffer) > self.config['feedback_buffer_size']:
                # 移除最旧的已处理反馈
                processed_feedbacks = [f for f in self.feedback_buffer if f['processed']]
                if processed_feedbacks:
                    oldest_processed = min(processed_feedbacks, key=lambda x: x['timestamp'])
                    self.feedback_buffer.remove(oldest_processed)
                else:
                    # 如果没有已处理的反馈，移除最旧的
                    self.feedback_buffer.pop(0)

            logger.debug(f"已添加反馈: {feedback_data.get('type', 'unknown')}")

        except Exception as e:
            logger.error(f"添加反馈失败: {e}")

    def _execute_learning_cycle(self):
        """执行学习周期"""
        try:
            cycle_start = datetime.now()

            # 检查是否有足够的反馈数据
            unprocessed_feedback = [f for f in self.feedback_buffer if not f['processed']]

            if len(unprocessed_feedback) < self.config['min_samples_for_learning']:
                logger.debug(f"反馈数据不足 ({len(unprocessed_feedback)}/{self.config['min_samples_for_learning']})，跳过学习周期")
                return

            logger.info(f"开始学习周期，处理 {len(unprocessed_feedback)} 条反馈数据")

            # 分析反馈数据
            feedback_analysis = self._analyze_feedback_data(unprocessed_feedback)

            # 生成学习计划
            learning_plan = self._generate_learning_plan(feedback_analysis)

            # 执行学习
            learning_results = self._execute_learning(learning_plan)

            # 评估学习效果
            evaluation_results = self._evaluate_learning_effectiveness(learning_results)

            # 记录学习周期
            cycle_record = {
                'cycle_id': f"learning_cycle_{int(cycle_start.timestamp())}",
                'start_time': cycle_start,
                'end_time': datetime.now(),
                'feedback_processed': len(unprocessed_feedback),
                'learning_plan': learning_plan,
                'learning_results': learning_results,
                'evaluation_results': evaluation_results,
                'status': 'completed' if evaluation_results.get('improvement_achieved', False) else 'no_improvement'
            }

            self.learning_cycles.append(cycle_record)

            # 标记反馈为已处理
            for feedback in unprocessed_feedback:
                feedback['processed'] = True
                feedback['processed_at'] = datetime.now()

            # 限制学习周期历史
            max_cycles = 100
            if len(self.learning_cycles) > max_cycles:
                self.learning_cycles = self.learning_cycles[-max_cycles:]

            logger.info(f"学习周期完成: {cycle_record['cycle_id']}")

        except Exception as e:
            logger.error(f"学习周期执行失败: {e}")

    def _analyze_feedback_data(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析反馈数据"""
        try:
            analysis = {
                'total_feedback': len(feedback_data),
                'feedback_types': {},
                'performance_feedback': [],
                'error_feedback': [],
                'user_feedback': [],
                'system_feedback': [],
                'temporal_patterns': {},
                'severity_distribution': {}
            }

            # 分类反馈
            for feedback in feedback_data:
                feedback_type = feedback.get('feedback_type', 'general')
                analysis['feedback_types'][feedback_type] = analysis['feedback_types'].get(feedback_type, 0) + 1

                data = feedback.get('data', {})

                # 分类不同类型的反馈
                if feedback_type == 'performance':
                    analysis['performance_feedback'].append(data)
                elif feedback_type == 'error':
                    analysis['error_feedback'].append(data)
                elif feedback_type == 'user':
                    analysis['user_feedback'].append(data)
                elif feedback_type == 'system':
                    analysis['system_feedback'].append(data)

                # 分析严重程度
                severity = data.get('severity', 'low')
                analysis['severity_distribution'][severity] = analysis['severity_distribution'].get(severity, 0) + 1

            # 分析时间模式
            timestamps = [f['timestamp'] for f in feedback_data]
            if timestamps:
                analysis['temporal_patterns'] = self._analyze_temporal_patterns(timestamps)

            return analysis

        except Exception as e:
            logger.error(f"反馈数据分析失败: {e}")
            return {'error': str(e)}

    def _analyze_temporal_patterns(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """分析时间模式"""
        try:
            # 转换为小时
            hours = [ts.hour for ts in timestamps]

            # 统计每个小时的反馈数量
            hour_distribution = {}
            for hour in range(24):
                hour_distribution[hour] = hours.count(hour)

            # 找出高峰时段
            peak_hours = [hour for hour, count in hour_distribution.items() if count > np.mean(list(hour_distribution.values())) * 1.5]

            return {
                'peak_hours': peak_hours,
                'total_period': f"{min(timestamps)} to {max(timestamps)}",
                'average_per_hour': len(timestamps) / 24,
                'hourly_distribution': hour_distribution
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_learning_plan(self, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成学习计划"""
        try:
            plan = {
                'learning_objectives': [],
                'data_sources': [],
                'algorithms_to_update': [],
                'expected_improvements': [],
                'risk_assessment': {},
                'timeline': {}
            }

            # 基于反馈分析生成学习目标
            if feedback_analysis.get('performance_feedback'):
                plan['learning_objectives'].append('Improve performance prediction accuracy')
                plan['algorithms_to_update'].append('performance_predictor')
                plan['expected_improvements'].append('Reduce prediction error by 10%')

            if feedback_analysis.get('error_feedback'):
                plan['learning_objectives'].append('Enhance error detection and classification')
                plan['algorithms_to_update'].append('anomaly_detector')
                plan['expected_improvements'].append('Increase error detection rate by 15%')

            if feedback_analysis.get('user_feedback'):
                plan['learning_objectives'].append('Better understand user behavior patterns')
                plan['algorithms_to_update'].append('quality_predictor')
                plan['expected_improvements'].append('Improve user satisfaction prediction by 20%')

            # 设置数据源
            plan['data_sources'] = ['feedback_buffer', 'system_metrics', 'quality_history']

            # 风险评估
            high_severity_count = feedback_analysis.get('severity_distribution', {}).get('high', 0)
            critical_severity_count = feedback_analysis.get('severity_distribution', {}).get('critical', 0)

            if high_severity_count > 10 or critical_severity_count > 5:
                plan['risk_assessment'] = {
                    'level': 'high',
                    'concerns': ['Potential system instability', 'High feedback volume'],
                    'mitigation': ['Implement gradual rollout', 'Increase monitoring']
                }
            else:
                plan['risk_assessment'] = {
                    'level': 'low',
                    'concerns': [],
                    'mitigation': []
                }

            # 时间线
            plan['timeline'] = {
                'data_collection': '2 hours',
                'model_training': '4 hours',
                'validation': '1 hour',
                'deployment': '30 minutes'
            }

            return plan

        except Exception as e:
            logger.error(f"学习计划生成失败: {e}")
            return {'error': str(e)}

    def _execute_learning(self, learning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习"""
        try:
            results = {
                'algorithms_updated': [],
                'performance_improvements': {},
                'new_patterns_learned': [],
                'training_metrics': {},
                'execution_time': 0
            }

            start_time = time.time()

            # 这里应该实现具体的学习执行逻辑
            # 目前返回模拟结果

            # 模拟算法更新
            for algorithm in learning_plan.get('algorithms_to_update', []):
                results['algorithms_updated'].append(algorithm)
                results['performance_improvements'][algorithm] = np.random.uniform(0.05, 0.15)

            # 模拟新模式学习
            results['new_patterns_learned'] = [
                'Improved error pattern recognition',
                'Enhanced performance prediction',
                'Better user behavior modeling'
            ]

            # 模拟训练指标
            results['training_metrics'] = {
                'accuracy': np.random.uniform(0.85, 0.95),
                'precision': np.random.uniform(0.80, 0.90),
                'recall': np.random.uniform(0.75, 0.85),
                'f1_score': np.random.uniform(0.82, 0.92)
            }

            results['execution_time'] = time.time() - start_time

            return results

        except Exception as e:
            logger.error(f"学习执行失败: {e}")
            return {'error': str(e)}

    def _evaluate_learning_effectiveness(self, learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估学习效果"""
        try:
            evaluation = {
                'improvement_achieved': False,
                'overall_improvement': 0.0,
                'algorithms_improved': [],
                'metrics_improved': [],
                'recommendations': []
            }

            # 检查性能改进
            performance_improvements = learning_results.get('performance_improvements', {})
            if performance_improvements:
                avg_improvement = np.mean(list(performance_improvements.values()))

                if avg_improvement >= self.config['performance_improvement_threshold']:
                    evaluation['improvement_achieved'] = True
                    evaluation['overall_improvement'] = avg_improvement

                    # 找出改进的算法
                    for algorithm, improvement in performance_improvements.items():
                        if improvement >= self.config['performance_improvement_threshold']:
                            evaluation['algorithms_improved'].append(algorithm)

            # 检查指标改进
            training_metrics = learning_results.get('training_metrics', {})
            for metric, value in training_metrics.items():
                if value > 0.8:  # 假设80%以上为良好
                    evaluation['metrics_improved'].append(metric)

            # 生成建议
            if evaluation['improvement_achieved']:
                evaluation['recommendations'].append('Deploy improved models to production')
                evaluation['recommendations'].append('Continue monitoring performance improvements')
            else:
                evaluation['recommendations'].append('Collect more training data')
                evaluation['recommendations'].append('Review learning algorithms and parameters')

            return evaluation

        except Exception as e:
            logger.error(f"学习效果评估失败: {e}")
            return {'error': str(e)}

    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        try:
            stats = {
                'learning_active': self.learning_active,
                'total_cycles': len(self.learning_cycles),
                'feedback_buffer_size': len(self.feedback_buffer),
                'unprocessed_feedback': len([f for f in self.feedback_buffer if not f['processed']])
            }

            if self.learning_cycles:
                recent_cycles = self.learning_cycles[-10:]  # 最近10个周期

                successful_cycles = len([c for c in recent_cycles if c['status'] == 'completed'])
                stats.update({
                    'recent_success_rate': successful_cycles / len(recent_cycles),
                    'avg_cycle_duration': np.mean([
                        (c['end_time'] - c['start_time']).total_seconds()
                        for c in recent_cycles
                    ]),
                    'total_feedback_processed': sum(c['feedback_processed'] for c in recent_cycles)
                })

            return stats

        except Exception as e:
            logger.error(f"获取学习统计失败: {e}")
            return {'error': str(e)}


class AdaptiveModelOptimizer:
    """自适应模型优化器"""

    def __init__(self, optimization_config: Dict[str, Any] = None):
        self.config = optimization_config or self._get_default_config()
        self.model_versions = {}
        self.optimization_history = []
        self.performance_baseline = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optimization_interval': 7200,  # 2小时优化一次
            'drift_detection_threshold': 0.1,  # 10%漂移检测阈值
            'min_improvement_threshold': 0.03,  # 3%最小改进阈值
            'max_optimization_attempts': 5,
            'rollback_enabled': True,
            'a_b_testing_enabled': True
        }

    def register_model_for_optimization(self, model_name: str, model_instance: Any,
                                      performance_metrics: Dict[str, Any]):
        """注册模型进行优化"""
        try:
            self.model_versions[model_name] = {
                'current_version': model_instance,
                'performance_baseline': performance_metrics.copy(),
                'optimization_attempts': 0,
                'last_optimization': None,
                'optimization_history': []
            }

            self.performance_baseline[model_name] = performance_metrics

            logger.info(f"模型 {model_name} 已注册进行自适应优化")

        except Exception as e:
            logger.error(f"注册模型优化失败: {e}")

    def check_optimization_needed(self, model_name: str, current_performance: Dict[str, Any]) -> bool:
        """检查是否需要优化"""
        try:
            if model_name not in self.model_versions:
                return False

            baseline = self.performance_baseline.get(model_name, {})
            if not baseline:
                return True  # 没有基准，建议优化

            # 检查性能漂移
            for metric, baseline_value in baseline.items():
                if metric in current_performance:
                    current_value = current_performance[metric]

                    # 对于准确率等指标，值越低需要优化
                    if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                        if baseline_value - current_value > self.config['drift_detection_threshold']:
                            logger.info(f"模型 {model_name} {metric} 性能漂移检测: {baseline_value:.3f} -> {current_value:.3f}")
                            return True

                    # 对于错误率等指标，值越高需要优化
                    elif metric in ['error_rate', 'false_positive_rate']:
                        if current_value - baseline_value > self.config['drift_detection_threshold']:
                            logger.info(f"模型 {model_name} {metric} 性能漂移检测: {baseline_value:.3f} -> {current_value:.3f}")
                            return True

            # 检查优化频率
            last_optimization = self.model_versions[model_name]['last_optimization']
            if last_optimization:
                time_since_optimization = (datetime.now() - last_optimization).total_seconds()
                if time_since_optimization < self.config['optimization_interval']:
                    return False

            # 检查优化尝试次数
            attempts = self.model_versions[model_name]['optimization_attempts']
            if attempts >= self.config['max_optimization_attempts']:
                logger.warning(f"模型 {model_name} 已达到最大优化尝试次数 ({attempts})")
                return False

            return True

        except Exception as e:
            logger.error(f"优化需求检查失败: {e}")
            return False

    async def optimize_model(self, model_name: str, current_performance: Dict[str, Any],
                           new_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """优化模型"""
        try:
            if model_name not in self.model_versions:
                return {'success': False, 'error': 'Model not registered'}

            model_info = self.model_versions[model_name]

            logger.info(f"开始优化模型: {model_name}")

            # 执行优化
            optimization_result = await self._perform_model_optimization(
                model_name, current_performance, new_training_data
            )

            # 记录优化历史
            optimization_record = {
                'timestamp': datetime.now(),
                'model_name': model_name,
                'current_performance': current_performance,
                'optimization_result': optimization_result,
                'status': 'success' if optimization_result.get('success', False) else 'failed'
            }

            model_info['optimization_history'].append(optimization_record)
            self.optimization_history.append(optimization_record)

            # 更新模型信息
            if optimization_result.get('success', False):
                model_info['last_optimization'] = datetime.now()
                model_info['optimization_attempts'] = 0  # 重置尝试次数

                # 如果有新的模型版本，更新基准性能
                if 'new_performance' in optimization_result:
                    self.performance_baseline[model_name] = optimization_result['new_performance']

            else:
                model_info['optimization_attempts'] += 1

            # 限制历史记录
            max_history = 50
            if len(model_info['optimization_history']) > max_history:
                model_info['optimization_history'] = model_info['optimization_history'][-max_history:]

            logger.info(f"模型优化完成: {model_name}")

            return optimization_result

        except Exception as e:
            logger.error(f"模型优化失败: {e}")
            return {'success': False, 'error': str(e)}

    async def _perform_model_optimization(self, model_name: str, current_performance: Dict[str, Any],
                                       new_training_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """执行模型优化"""
        try:
            # 这里应该实现具体的模型优化逻辑
            # 目前返回模拟结果

            optimization_result = {
                'success': True,
                'optimization_method': 'incremental_learning',
                'parameters_adjusted': ['learning_rate', 'batch_size'],
                'new_performance': {
                    'accuracy': current_performance.get('accuracy', 0) + 0.05,
                    'precision': current_performance.get('precision', 0) + 0.03,
                    'recall': current_performance.get('recall', 0) + 0.04,
                    'error_rate': max(0, current_performance.get('error_rate', 0) - 0.02)
                },
                'improvement_metrics': {
                    'accuracy_improvement': 0.05,
                    'overall_improvement': 0.035
                },
                'training_time': 180.5,  # 秒
                'data_used': len(new_training_data) if new_training_data is not None else 0
            }

            return optimization_result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_optimization_stats(self, model_name: str = None) -> Dict[str, Any]:
        """获取优化统计"""
        try:
            if model_name:
                if model_name not in self.model_versions:
                    return {'error': 'Model not registered'}

                model_info = self.model_versions[model_name]
                history = model_info['optimization_history']

                successful_optimizations = len([h for h in history if h['status'] == 'success'])

                return {
                    'model_name': model_name,
                    'total_optimizations': len(history),
                    'successful_optimizations': successful_optimizations,
                    'success_rate': successful_optimizations / len(history) if history else 0,
                    'last_optimization': model_info['last_optimization'].isoformat() if model_info['last_optimization'] else None,
                    'optimization_attempts': model_info['optimization_attempts'],
                    'performance_baseline': self.performance_baseline.get(model_name, {})
                }
            else:
                # 全局统计
                total_optimizations = len(self.optimization_history)
                successful_optimizations = len([h for h in self.optimization_history if h['status'] == 'success'])

                return {
                    'total_models': len(self.model_versions),
                    'total_optimizations': total_optimizations,
                    'successful_optimizations': successful_optimizations,
                    'overall_success_rate': successful_optimizations / total_optimizations if total_optimizations > 0 else 0,
                    'models_optimized': list(self.model_versions.keys())
                }

        except Exception as e:
            logger.error(f"获取优化统计失败: {e}")
            return {'error': str(e)}


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_categories = {}
        self.knowledge_index = {}
        self.knowledge_stats = {
            'total_entries': 0,
            'categories': {},
            'last_updated': None,
            'quality_score': 0.0
        }

    def add_knowledge_entry(self, category: str, entry_id: str,
                          knowledge_data: Dict[str, Any]) -> bool:
        """添加知识条目"""
        try:
            # 初始化类别
            if category not in self.knowledge_categories:
                self.knowledge_categories[category] = {}

            # 添加条目
            entry_data = {
                'entry_id': entry_id,
                'category': category,
                'content': knowledge_data,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'access_count': 0,
                'usefulness_score': 0.0,
                'tags': knowledge_data.get('tags', [])
            }

            self.knowledge_categories[category][entry_id] = entry_data
            self.knowledge_index[entry_id] = category

            # 更新统计
            self.knowledge_stats['total_entries'] += 1
            self.knowledge_stats['categories'][category] = self.knowledge_stats['categories'].get(category, 0) + 1
            self.knowledge_stats['last_updated'] = datetime.now()

            # 保存到文件
            self._save_knowledge_entry(category, entry_id, entry_data)

            logger.info(f"知识条目已添加: {category}/{entry_id}")

            return True

        except Exception as e:
            logger.error(f"添加知识条目失败: {e}")
            return False

    def search_knowledge(self, query: str, category: str = None,
                        tags: List[str] = None) -> List[Dict[str, Any]]:
        """搜索知识"""
        try:
            results = []

            # 确定搜索范围
            categories_to_search = [category] if category else list(self.knowledge_categories.keys())

            for cat in categories_to_search:
                if cat not in self.knowledge_categories:
                    continue

                for entry_id, entry_data in self.knowledge_categories[cat].items():
                    # 检查标签匹配
                    if tags:
                        entry_tags = set(entry_data.get('tags', []))
                        query_tags = set(tags)
                        if not query_tags.issubset(entry_tags):
                            continue

                    # 检查内容匹配
                    content = entry_data.get('content', {})
                    content_str = json.dumps(content, ensure_ascii=False).lower()

                    if query.lower() in content_str:
                        # 计算相关性分数
                        relevance_score = self._calculate_relevance_score(query, entry_data)

                        result = entry_data.copy()
                        result['relevance_score'] = relevance_score

                        results.append(result)

                        # 增加访问计数
                        entry_data['access_count'] += 1

            # 按相关性排序
            results.sort(key=lambda x: x['relevance_score'], reverse=True)

            return results[:20]  # 返回前20个结果

        except Exception as e:
            logger.error(f"知识搜索失败: {e}")
            return []

    def _calculate_relevance_score(self, query: str, entry_data: Dict[str, Any]) -> float:
        """计算相关性分数"""
        try:
            score = 0.0

            # 查询匹配分数
            content = entry_data.get('content', {})
            content_str = json.dumps(content, ensure_ascii=False).lower()
            query_lower = query.lower()

            # 精确匹配
            if query_lower in content_str:
                score += 1.0

            # 词频匹配
            query_words = query_lower.split()
            for word in query_words:
                if word in content_str:
                    score += 0.5

            # 使用频率分数
            access_count = entry_data.get('access_count', 0)
            score += min(access_count * 0.1, 1.0)  # 最多加1分

            # 有用性分数
            usefulness = entry_data.get('usefulness_score', 0.0)
            score += usefulness * 0.5

            return score

        except Exception:
            return 0.0

    def update_knowledge_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识条目"""
        try:
            if entry_id not in self.knowledge_index:
                return False

            category = self.knowledge_index[entry_id]
            entry_data = self.knowledge_categories[category][entry_id]

            # 应用更新
            for key, value in updates.items():
                if key == 'usefulness_score':
                    # 更新有用性分数（可能是用户反馈）
                    current_score = entry_data.get('usefulness_score', 0.0)
                    new_score = (current_score + value) / 2  # 平均值
                    entry_data['usefulness_score'] = new_score
                else:
                    entry_data['content'][key] = value

            entry_data['updated_at'] = datetime.now()

            # 保存更新
            self._save_knowledge_entry(category, entry_id, entry_data)

            return True

        except Exception as e:
            logger.error(f"更新知识条目失败: {e}")
            return False

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        try:
            stats = self.knowledge_stats.copy()

            # 计算质量分数
            total_entries = stats['total_entries']
            if total_entries > 0:
                # 基于访问频率和有用性计算质量
                total_access = 0
                total_usefulness = 0

                for category_entries in self.knowledge_categories.values():
                    for entry_data in category_entries.values():
                        total_access += entry_data.get('access_count', 0)
                        total_usefulness += entry_data.get('usefulness_score', 0.0)

                avg_access = total_access / total_entries
                avg_usefulness = total_usefulness / total_entries

                # 质量分数 = 访问率 * 0.6 + 有用性 * 0.4
                quality_score = min(1.0, (avg_access * 0.01) * 0.6 + avg_usefulness * 0.4)
                stats['quality_score'] = quality_score

            return stats

        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            return {'error': str(e)}

    def _save_knowledge_entry(self, category: str, entry_id: str, entry_data: Dict[str, Any]):
        """保存知识条目到文件"""
        try:
            category_path = self.kb_path / category
            category_path.mkdir(exist_ok=True)

            entry_file = category_path / f"{entry_id}.json"
            with open(entry_file, 'w', encoding='utf-8') as f:
                json.dump(entry_data, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            logger.error(f"保存知识条目失败: {e}")

    def cleanup_old_knowledge(self, max_age_days: int = 365) -> int:
        """清理旧知识"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0

            for category, entries in self.knowledge_categories.items():
                entries_to_remove = []

                for entry_id, entry_data in entries.items():
                    if entry_data['updated_at'] < cutoff_date:
                        entries_to_remove.append(entry_id)

                # 移除旧条目
                for entry_id in entries_to_remove:
                    del entries[entry_id]
                    del self.knowledge_index[entry_id]

                    # 删除文件
                    entry_file = self.kb_path / category / f"{entry_id}.json"
                    if entry_file.exists():
                        entry_file.unlink()

                    removed_count += 1

            # 更新统计
            self.knowledge_stats['total_entries'] -= removed_count
            for category in self.knowledge_categories:
                self.knowledge_stats['categories'][category] = len(self.knowledge_categories[category])

            logger.info(f"已清理 {removed_count} 条旧知识")

            return removed_count

        except Exception as e:
            logger.error(f"清理旧知识失败: {e}")
            return 0


class PerformanceTrendLearner:
    """性能趋势学习器"""

    def __init__(self):
        self.trend_models = {}
        self.pattern_library = {}
        self.prediction_history = []

    def learn_performance_trends(self, historical_data: pd.DataFrame,
                                performance_metrics: List[str]) -> Dict[str, Any]:
        """学习性能趋势"""
        try:
            learning_results = {}

            for metric in performance_metrics:
                if metric in historical_data.columns:
                    # 学习该指标的趋势
                    trend_model = self._learn_metric_trend(historical_data, metric)
                    self.trend_models[metric] = trend_model

                    # 识别模式
                    patterns = self._identify_metric_patterns(historical_data, metric)
                    self.pattern_library[metric] = patterns

                    learning_results[metric] = {
                        'trend_model': trend_model,
                        'patterns': patterns,
                        'confidence': self._calculate_trend_confidence(trend_model, patterns)
                    }

            return learning_results

        except Exception as e:
            logger.error(f"性能趋势学习失败: {e}")
            return {'error': str(e)}

    def _learn_metric_trend(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """学习指标趋势"""
        try:
            values = data[metric].values
            timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else range(len(values))

            # 使用线性回归学习趋势
            from sklearn.linear_model import LinearRegression

            X = np.arange(len(values)).reshape(-1, 1)
            y = values

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0]
            intercept = model.intercept_

            # 确定趋势类型
            if slope > 0.001:
                trend_type = 'increasing'
            elif slope < -0.001:
                trend_type = 'decreasing'
            else:
                trend_type = 'stable'

            return {
                'slope': slope,
                'intercept': intercept,
                'trend_type': trend_type,
                'r_squared': model.score(X, y),
                'predictions': model.predict(X)
            }

        except Exception as e:
            logger.error(f"指标趋势学习失败 {metric}: {e}")
            return {'error': str(e)}

    def _identify_metric_patterns(self, data: pd.DataFrame, metric: str) -> List[Dict[str, Any]]:
        """识别指标模式"""
        try:
            values = data[metric].values
            patterns = []

            # 检测周期性模式
            from scipy.signal import find_peaks

            # 寻找峰值
            peaks, _ = find_peaks(values, distance=len(values)//10)
            if len(peaks) > 2:
                avg_distance = np.mean(np.diff(peaks))
                patterns.append({
                    'type': 'periodic',
                    'period': avg_distance,
                    'confidence': min(0.8, len(peaks) / 10)
                })

            # 检测季节性模式（如果有时间戳）
            if isinstance(data.index, pd.DatetimeIndex):
                hourly_avg = data.groupby(data.index.hour)[metric].mean()
                if hourly_avg.std() > hourly_avg.mean() * 0.1:
                    patterns.append({
                        'type': 'hourly_seasonal',
                        'peak_hour': hourly_avg.idxmax(),
                        'confidence': 0.7
                    })

            # 检测异常模式
            mean_val = np.mean(values)
            std_val = np.std(values)
            anomalies = [i for i, v in enumerate(values) if abs(v - mean_val) > 3 * std_val]

            if anomalies:
                patterns.append({
                    'type': 'anomalies',
                    'anomaly_count': len(anomalies),
                    'anomaly_indices': anomalies[:10],  # 只记录前10个
                    'confidence': min(0.9, len(anomalies) / len(values))
                })

            return patterns

        except Exception as e:
            logger.error(f"指标模式识别失败 {metric}: {e}")
            return []

    def _calculate_trend_confidence(self, trend_model: Dict[str, Any],
                                  patterns: List[Dict[str, Any]]) -> float:
        """计算趋势置信度"""
        try:
            confidence = 0.5  # 基础置信度

            # 基于R²分数
            r_squared = trend_model.get('r_squared', 0)
            confidence += r_squared * 0.3

            # 基于模式识别
            if patterns:
                pattern_confidence = np.mean([p.get('confidence', 0.5) for p in patterns])
                confidence += pattern_confidence * 0.2

            return min(0.95, confidence)

        except Exception:
            return 0.5

    def predict_future_performance(self, metric: str, prediction_horizon: int = 24) -> Dict[str, Any]:
        """预测未来性能"""
        try:
            if metric not in self.trend_models:
                return {'error': 'No trend model available'}

            trend_model = self.trend_models[metric]

            # 基于趋势模型进行预测
            last_index = 100  # 假设当前是第100个时间点
            future_indices = np.arange(last_index, last_index + prediction_horizon).reshape(-1, 1)

            predictions = trend_model['slope'] * future_indices.flatten() + trend_model['intercept']

            # 应用模式调整
            if metric in self.pattern_library:
                patterns = self.pattern_library[metric]
                predictions = self._apply_pattern_adjustments(predictions, patterns)

            return {
                'metric': metric,
                'predictions': predictions.tolist(),
                'trend_type': trend_model['trend_type'],
                'confidence': trend_model.get('r_squared', 0.5),
                'prediction_horizon': prediction_horizon
            }

        except Exception as e:
            logger.error(f"未来性能预测失败 {metric}: {e}")
            return {'error': str(e)}

    def _apply_pattern_adjustments(self, predictions: np.ndarray,
                                 patterns: List[Dict[str, Any]]) -> np.ndarray:
        """应用模式调整"""
        try:
            adjusted_predictions = predictions.copy()

            for pattern in patterns:
                if pattern['type'] == 'periodic':
                    # 应用周期性调整
                    period = pattern.get('period', len(predictions))
                    amplitude = np.std(predictions) * 0.1  # 10%的振幅

                    for i in range(len(predictions)):
                        phase = (i % period) / period * 2 * np.pi
                        adjusted_predictions[i] += amplitude * np.sin(phase)

                elif pattern['type'] == 'hourly_seasonal':
                    # 应用小时季节性调整
                    peak_hour = pattern.get('peak_hour', 12)
                    for i in range(len(predictions)):
                        hour = i % 24
                        distance_from_peak = min(abs(hour - peak_hour), 24 - abs(hour - peak_hour))
                        seasonal_factor = 1 + 0.1 * np.exp(-distance_from_peak / 3)  # 高斯衰减
                        adjusted_predictions[i] *= seasonal_factor

            return adjusted_predictions

        except Exception as e:
            logger.error(f"模式调整失败: {e}")
            return predictions


class ContinuousOptimizationManager:
    """连续优化管理器"""

    def __init__(self):
        self.learning_manager = LearningLoopManager()
        self.model_optimizer = AdaptiveModelOptimizer()
        self.knowledge_manager = KnowledgeBaseManager()
        self.trend_learner = PerformanceTrendLearner()
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'knowledge_entries': 0,
            'trend_models': 0,
            'last_optimization': None
        }

    async def initialize_continuous_optimization(self):
        """初始化连续优化"""
        try:
            # 启动学习循环
            self.learning_manager.start_learning_loop()

            logger.info("连续优化管理器初始化完成")

        except Exception as e:
            logger.error(f"连续优化初始化失败: {e}")

    def add_optimization_feedback(self, feedback_data: Dict[str, Any]):
        """添加优化反馈"""
        try:
            self.learning_manager.add_feedback(feedback_data)

            # 如果是知识相关的反馈，添加到知识库
            if 'knowledge_category' in feedback_data:
                self.knowledge_manager.add_knowledge_entry(
                    feedback_data['knowledge_category'],
                    f"feedback_{int(datetime.now().timestamp())}",
                    feedback_data
                )

        except Exception as e:
            logger.error(f"添加优化反馈失败: {e}")

    async def perform_continuous_optimization(self, system_metrics: Dict[str, Any],
                                           performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行连续优化"""
        try:
            optimization_results = {
                'timestamp': datetime.now(),
                'learning_cycle': {},
                'model_optimization': {},
                'knowledge_update': {},
                'trend_learning': {},
                'overall_improvement': 0.0
            }

            # 执行学习周期
            learning_stats = self.learning_manager.get_learning_stats()
            optimization_results['learning_cycle'] = learning_stats

            # 检查模型优化需求
            models_needing_optimization = []
            for model_name in ['anomaly_detector', 'quality_predictor', 'performance_predictor']:
                if self.model_optimizer.check_optimization_needed(model_name, performance_data):
                    models_needing_optimization.append(model_name)

            # 执行模型优化
            if models_needing_optimization:
                for model_name in models_needing_optimization:
                    opt_result = await self.model_optimizer.optimize_model(
                        model_name, performance_data
                    )
                    optimization_results['model_optimization'][model_name] = opt_result

                    if opt_result.get('success', False):
                        optimization_results['overall_improvement'] += opt_result.get('improvement_metrics', {}).get('overall_improvement', 0)

            # 更新知识库
            knowledge_stats = self.knowledge_manager.get_knowledge_stats()
            optimization_results['knowledge_update'] = knowledge_stats

            # 学习性能趋势
            if 'performance_history' in system_metrics:
                perf_history = pd.DataFrame(system_metrics['performance_history'])
                trend_results = self.trend_learner.learn_performance_trends(
                    perf_history, ['response_time', 'error_rate', 'throughput']
                )
                optimization_results['trend_learning'] = trend_results

            # 更新统计
            self.optimization_stats['total_optimizations'] += 1
            if optimization_results['overall_improvement'] > 0:
                self.optimization_stats['successful_optimizations'] += 1
            self.optimization_stats['knowledge_entries'] = knowledge_stats.get('total_entries', 0)
            self.optimization_stats['trend_models'] = len(self.trend_learner.trend_models)
            self.optimization_stats['last_optimization'] = datetime.now()

            return optimization_results

        except Exception as e:
            logger.error(f"连续优化执行失败: {e}")
            return {'error': str(e)}

    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """获取优化仪表板"""
        try:
            dashboard = {
                'timestamp': datetime.now(),
                'optimization_stats': self.optimization_stats.copy(),
                'learning_stats': self.learning_manager.get_learning_stats(),
                'model_optimization_stats': self.model_optimizer.get_optimization_stats(),
                'knowledge_stats': self.knowledge_manager.get_knowledge_stats(),
                'active_trend_models': list(self.trend_learner.trend_models.keys()),
                'system_health': 'optimizing' if self.learning_manager.learning_active else 'idle'
            }

            # 计算优化效率
            total_opts = dashboard['optimization_stats']['total_optimizations']
            successful_opts = dashboard['optimization_stats']['successful_optimizations']

            dashboard['optimization_efficiency'] = {
                'success_rate': successful_opts / total_opts if total_opts > 0 else 0,
                'avg_improvement': 0.05,  # 模拟值，实际应该计算
                'knowledge_growth_rate': dashboard['knowledge_stats'].get('total_entries', 0) / max(total_opts, 1)
            }

            return dashboard

        except Exception as e:
            logger.error(f"获取优化仪表板失败: {e}")
            return {'error': str(e)}
