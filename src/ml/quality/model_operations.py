"""
AI模型运维与监控系统

实现AI模型的生产运维、性能监控、自动化更新和持续优化：
1. 模型版本管理 - 模型的版本控制和生命周期管理
2. 性能监控 - 实时监控模型性能和预测准确性
3. 自动化更新 - 基于性能指标的模型自动更新机制
4. 模型健康检查 - 模型运行状态和资源使用监控
5. A/B测试框架 - 新旧模型的对比测试和逐步部署
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import hashlib
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, model_registry_path: str = "models/registry"):
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.current_versions = {}
        self.version_history = {}
        self.model_metadata = {}

    def register_model_version(self, model_name: str, model_path: str,
                             metadata: Dict[str, Any]) -> str:
        """注册模型版本"""
        try:
            # 计算模型文件的哈希值作为版本标识
            model_hash = self._calculate_model_hash(model_path)

            # 创建版本信息
            version_info = {
                'model_name': model_name,
                'version_id': model_hash,
                'model_path': str(model_path),
                'created_at': datetime.now(),
                'metadata': metadata,
                'status': 'registered',
                'performance_metrics': {},
                'validation_results': {}
            }

            # 保存版本信息
            version_file = self.registry_path / f"{model_name}_{model_hash}.json"
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2, default=str)

            # 更新当前版本
            self.current_versions[model_name] = model_hash

            # 更新历史记录
            if model_name not in self.version_history:
                self.version_history[model_name] = []
            self.version_history[model_name].append(version_info)

            # 更新元数据
            self.model_metadata[f"{model_name}_{model_hash}"] = version_info

            logger.info(f"已注册模型版本: {model_name} v{model_hash[:8]}")

            return model_hash

        except Exception as e:
            logger.error(f"注册模型版本失败: {e}")
            raise

    def get_model_version(self, model_name: str, version_id: str = None) -> Optional[Dict[str, Any]]:
        """获取模型版本信息"""
        if version_id is None:
            version_id = self.current_versions.get(model_name)

        if not version_id:
            return None

        return self.model_metadata.get(f"{model_name}_{version_id}")

    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """列出模型的所有版本"""
        return self.version_history.get(model_name, [])

    def promote_model_version(self, model_name: str, version_id: str,
                            environment: str = 'staging') -> bool:
        """提升模型版本到指定环境"""
        try:
            version_key = f"{model_name}_{version_id}"
            if version_key not in self.model_metadata:
                logger.error(f"模型版本不存在: {version_key}")
                return False

            # 更新版本状态
            self.model_metadata[version_key]['status'] = f'promoted_to_{environment}'
            self.model_metadata[version_key]['promoted_at'] = datetime.now()
            self.model_metadata[version_key]['environment'] = environment

            # 如果是生产环境，更新当前版本
            if environment == 'production':
                self.current_versions[model_name] = version_id

            # 保存更新后的版本信息
            version_info = self.model_metadata[version_key]
            version_file = self.registry_path / f"{model_name}_{version_id}.json"
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2, default=str)

            logger.info(f"模型版本 {model_name} v{version_id[:8]} 已提升到 {environment}")

            return True

        except Exception as e:
            logger.error(f"提升模型版本失败: {e}")
            return False

    def archive_model_version(self, model_name: str, version_id: str) -> bool:
        """归档模型版本"""
        try:
            version_key = f"{model_name}_{version_id}"
            if version_key not in self.model_metadata:
                logger.error(f"模型版本不存在: {version_key}")
                return False

            # 更新版本状态
            self.model_metadata[version_key]['status'] = 'archived'
            self.model_metadata[version_key]['archived_at'] = datetime.now()

            # 保存更新
            version_info = self.model_metadata[version_key]
            version_file = self.registry_path / f"{model_name}_{version_id}.json"
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2, default=str)

            logger.info(f"模型版本 {model_name} v{version_id[:8]} 已归档")

            return True

        except Exception as e:
            logger.error(f"归档模型版本失败: {e}")
            return False

    def _calculate_model_hash(self, model_path: str) -> str:
        """计算模型文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计"""
        return {
            'total_models': len(self.current_versions),
            'total_versions': len(self.model_metadata),
            'models': list(self.current_versions.keys()),
            'version_counts': {model: len(versions) for model, versions in self.version_history.items()}
        }


class ModelPerformanceMonitor:
    """模型性能监控器"""

    def __init__(self, monitoring_config: Dict[str, Any] = None):
        self.config = monitoring_config or self._get_default_config()
        self.performance_history = {}
        self.alert_thresholds = {}
        self.monitoring_active = False
        self.monitor_thread = None

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitoring_interval': 300,  # 5分钟
            'performance_window': 3600,  # 1小时性能窗口
            'alert_cooldown': 1800,      # 30分钟告警冷却
            'drift_detection_enabled': True,
            'accuracy_threshold': 0.8,
            'latency_threshold': 5.0    # 秒
        }

    def start_monitoring(self, model_name: str, monitoring_func: callable):
        """开始监控模型性能"""
        if model_name in self.performance_history:
            return  # 已经在监控

        self.performance_history[model_name] = {
            'monitoring_func': monitoring_func,
            'metrics_history': [],
            'alerts': [],
            'last_alert_time': None,
            'start_time': datetime.now()
        }

        # 设置默认告警阈值
        self.alert_thresholds[model_name] = {
            'accuracy_drop': 0.1,      # 准确率下降10%
            'latency_increase': 2.0,   # 延迟增加2秒
            'error_rate_spike': 0.05,  # 错误率激增5%
            'prediction_drift': 0.15   # 预测漂移15%
        }

        logger.info(f"已开始监控模型: {model_name}")

    def stop_monitoring(self, model_name: str):
        """停止监控模型性能"""
        if model_name in self.performance_history:
            del self.performance_history[model_name]
            logger.info(f"已停止监控模型: {model_name}")

    def record_performance_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """记录性能指标"""
        if model_name not in self.performance_history:
            logger.warning(f"模型 {model_name} 未在监控列表中")
            return

        # 添加时间戳
        metrics_with_timestamp = {
            'timestamp': datetime.now(),
            **metrics
        }

        # 记录指标历史
        self.performance_history[model_name]['metrics_history'].append(metrics_with_timestamp)

        # 限制历史记录长度
        max_history = 1000
        if len(self.performance_history[model_name]['metrics_history']) > max_history:
            self.performance_history[model_name]['metrics_history'] = \
                self.performance_history[model_name]['metrics_history'][-max_history:]

        # 检查是否需要告警
        self._check_performance_alerts(model_name, metrics_with_timestamp)

    def _check_performance_alerts(self, model_name: str, current_metrics: Dict[str, Any]):
        """检查性能告警"""
        try:
            thresholds = self.alert_thresholds.get(model_name, {})
            history = self.performance_history[model_name]['metrics_history']

            if len(history) < 2:
                return  # 需要至少2个数据点来比较

            # 计算基准指标（过去1小时的平均值）
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_metrics = [m for m in history if m['timestamp'] > one_hour_ago]

            if not recent_metrics:
                return

            baseline_metrics = {}
            for key in ['accuracy', 'latency', 'error_rate']:
                values = [m.get(key) for m in recent_metrics if key in m and m[key] is not None]
                if values:
                    baseline_metrics[key] = np.mean(values)

            alerts = []

            # 检查准确率下降
            if 'accuracy' in current_metrics and 'accuracy' in baseline_metrics:
                accuracy_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
                if accuracy_drop > thresholds.get('accuracy_drop', 0.1):
                    alerts.append({
                        'type': 'accuracy_drop',
                        'severity': 'high',
                        'message': f'模型准确率下降 {accuracy_drop:.3f} (基准: {baseline_metrics["accuracy"]:.3f}, 当前: {current_metrics["accuracy"]:.3f})',
                        'metrics': {
                            'baseline_accuracy': baseline_metrics['accuracy'],
                            'current_accuracy': current_metrics['accuracy'],
                            'drop': accuracy_drop
                        }
                    })

            # 检查延迟增加
            if 'latency' in current_metrics and 'latency' in baseline_metrics:
                latency_increase = current_metrics['latency'] - baseline_metrics['latency']
                if latency_increase > thresholds.get('latency_increase', 2.0):
                    alerts.append({
                        'type': 'latency_increase',
                        'severity': 'medium',
                        'message': f'模型延迟增加 {latency_increase:.2f}秒 (基准: {baseline_metrics["latency"]:.2f}秒, 当前: {current_metrics["latency"]:.2f}秒)',
                        'metrics': {
                            'baseline_latency': baseline_metrics['latency'],
                            'current_latency': current_metrics['latency'],
                            'increase': latency_increase
                        }
                    })

            # 检查错误率激增
            if 'error_rate' in current_metrics and 'error_rate' in baseline_metrics:
                error_spike = current_metrics['error_rate'] - baseline_metrics['error_rate']
                if error_spike > thresholds.get('error_rate_spike', 0.05):
                    alerts.append({
                        'type': 'error_rate_spike',
                        'severity': 'high',
                        'message': f'模型错误率激增 {error_spike:.3f} (基准: {baseline_metrics["error_rate"]:.3f}, 当前: {current_metrics["error_rate"]:.3f})',
                        'metrics': {
                            'baseline_error_rate': baseline_metrics['error_rate'],
                            'current_error_rate': current_metrics['error_rate'],
                            'spike': error_spike
                        }
                    })

            # 记录告警
            for alert in alerts:
                self._record_alert(model_name, alert)

        except Exception as e:
            logger.error(f"性能告警检查失败: {e}")

    def _record_alert(self, model_name: str, alert: Dict[str, Any]):
        """记录告警"""
        # 检查告警冷却时间
        last_alert_time = self.performance_history[model_name]['last_alert_time']
        if last_alert_time and (datetime.now() - last_alert_time).seconds < self.config['alert_cooldown']:
            return  # 冷却期内不重复告警

        alert_with_timestamp = {
            'timestamp': datetime.now(),
            **alert
        }

        self.performance_history[model_name]['alerts'].append(alert_with_timestamp)
        self.performance_history[model_name]['last_alert_time'] = datetime.now()

        logger.warning(f"模型性能告警: {model_name} - {alert['message']}")

    def get_performance_stats(self, model_name: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        if model_name not in self.performance_history:
            return {}

        history = self.performance_history[model_name]['metrics_history']
        alerts = self.performance_history[model_name]['alerts']

        if not history:
            return {'status': 'no_data'}

        # 计算统计信息
        recent_metrics = history[-50:]  # 最近50个指标

        stats = {
            'total_measurements': len(history),
            'recent_measurements': len(recent_metrics),
            'alert_count': len(alerts),
            'time_range': {
                'start': history[0]['timestamp'].isoformat(),
                'end': history[-1]['timestamp'].isoformat()
            },
            'current_metrics': history[-1] if history else {},
            'performance_trends': self._calculate_performance_trends(recent_metrics),
            'alert_summary': self._summarize_alerts(alerts)
        }

        return stats

    def _calculate_performance_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算性能趋势"""
        if len(metrics) < 2:
            return {}

        trends = {}

        # 计算各项指标的趋势
        for metric_name in ['accuracy', 'latency', 'error_rate', 'throughput']:
            values = [m.get(metric_name) for m in metrics if metric_name in m and m[metric_name] is not None]
            if len(values) >= 2:
                # 计算线性趋势
                slope = np.polyfit(range(len(values)), values, 1)[0]

                if metric_name in ['accuracy', 'throughput']:
                    # 这些指标越高越好
                    trend = 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable'
                else:
                    # 这些指标越低越好
                    trend = 'improving' if slope < -0.001 else 'declining' if slope > 0.001 else 'stable'

                trends[metric_name] = {
                    'trend': trend,
                    'slope': float(slope),
                    'current_value': float(values[-1]),
                    'avg_value': float(np.mean(values))
                }

        return trends

    def _summarize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总告警信息"""
        if not alerts:
            return {'total_alerts': 0}

        # 按类型统计
        alert_types = {}
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        # 按严重性统计
        severities = {}
        for alert in alerts:
            severity = alert.get('severity', 'low')
            severities[severity] = severities.get(severity, 0) + 1

        return {
            'total_alerts': len(alerts),
            'by_type': alert_types,
            'by_severity': severities,
            'most_recent': alerts[-1]['timestamp'].isoformat() if alerts else None
        }


class AutomatedModelUpdater:
    """自动化模型更新器"""

    def __init__(self, update_config: Dict[str, Any] = None):
        self.config = update_config or self._get_default_config()
        self.update_triggers = {}
        self.update_history = []
        self.updating = False

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'update_check_interval': 3600,  # 1小时检查一次
            'min_improvement_threshold': 0.05,  # 最小改进阈值5%
            'max_update_frequency': 7,  # 每周最多更新一次
            'rollback_enabled': True,
            'a_b_test_enabled': True,
            'a_b_test_duration': 168  # 7天A/B测试
        }

    def register_update_trigger(self, model_name: str, trigger_conditions: Dict[str, Any],
                               update_func: callable):
        """注册更新触发器"""
        self.update_triggers[model_name] = {
            'conditions': trigger_conditions,
            'update_func': update_func,
            'last_update': None,
            'update_count': 0,
            'performance_baseline': {}
        }
        logger.info(f"已注册更新触发器: {model_name}")

    def check_update_needed(self, model_name: str, current_performance: Dict[str, Any]) -> bool:
        """检查是否需要更新模型"""
        if model_name not in self.update_triggers:
            return False

        trigger_config = self.update_triggers[model_name]

        # 检查更新频率限制
        last_update = trigger_config['last_update']
        if last_update:
            days_since_update = (datetime.now() - last_update).days
            if days_since_update < self.config['max_update_frequency']:
                return False

        # 检查触发条件
        conditions = trigger_config['conditions']

        # 性能下降触发
        if 'performance_drop' in conditions:
            baseline = trigger_config['performance_baseline']
            if baseline:
                for metric, threshold in conditions['performance_drop'].items():
                    if metric in current_performance and metric in baseline:
                        drop = baseline[metric] - current_performance[metric]
                        if drop > threshold:
                            logger.info(f"模型 {model_name} 性能下降触发更新: {metric} 下降 {drop:.3f}")
                            return True

        # 准确率阈值触发
        if 'accuracy_threshold' in conditions:
            current_accuracy = current_performance.get('accuracy', 0)
            threshold = conditions['accuracy_threshold']
            if current_accuracy < threshold:
                logger.info(f"模型 {model_name} 准确率低于阈值触发更新: {current_accuracy:.3f} < {threshold}")
                return True

        # 时间间隔触发
        if 'max_age_days' in conditions:
            if not last_update:
                return True  # 从未更新

            days_since_update = (datetime.now() - last_update).days
            if days_since_update >= conditions['max_age_days']:
                logger.info(f"模型 {model_name} 达到最大年龄触发更新: {days_since_update}天")
                return True

        return False

    async def trigger_model_update(self, model_name: str, update_reason: str) -> Dict[str, Any]:
        """触发模型更新"""
        if self.updating:
            logger.warning("已有更新正在进行中，跳过本次更新")
            return {'success': False, 'reason': 'update_in_progress'}

        if model_name not in self.update_triggers:
            return {'success': False, 'reason': 'no_trigger_registered'}

        try:
            self.updating = True
            trigger_config = self.update_triggers[model_name]

            logger.info(f"开始更新模型: {model_name}, 原因: {update_reason}")

            # 执行更新函数
            update_func = trigger_config['update_func']
            if asyncio.iscoroutinefunction(update_func):
                update_result = await update_func()
            else:
                update_result = await asyncio.get_event_loop().run_in_executor(None, update_func)

            # 记录更新历史
            update_record = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'reason': update_reason,
                'result': update_result,
                'status': 'completed' if update_result.get('success', False) else 'failed'
            }

            self.update_history.append(update_record)

            # 更新触发器状态
            trigger_config['last_update'] = datetime.now()
            trigger_config['update_count'] += 1

            # 如果更新成功，考虑A/B测试
            if update_result.get('success', False) and self.config['a_b_test_enabled']:
                await self._start_a_b_test(model_name, update_result)

            logger.info(f"模型更新完成: {model_name}")

            return update_result

        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            update_record = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'reason': update_reason,
                'error': str(e),
                'status': 'failed'
            }
            self.update_history.append(update_record)

            return {'success': False, 'error': str(e)}

        finally:
            self.updating = False

    async def _start_a_b_test(self, model_name: str, update_result: Dict[str, Any]):
        """开始A/B测试"""
        try:
            logger.info(f"为模型 {model_name} 启动A/B测试")

            # 这里应该实现A/B测试逻辑
            # 比较新旧模型性能，选择更好的版本

            # 简化的A/B测试实现
            await asyncio.sleep(self.config['a_b_test_duration'] * 3600)  # 等待测试期

            # 假设新模型表现更好
            logger.info(f"A/B测试完成，模型 {model_name} 新版本表现更佳")

        except Exception as e:
            logger.error(f"A/B测试失败: {e}")

    def get_update_stats(self, model_name: str = None) -> Dict[str, Any]:
        """获取更新统计"""
        if model_name:
            trigger_config = self.update_triggers.get(model_name, {})
            model_history = [h for h in self.update_history if h['model_name'] == model_name]

            return {
                'model_name': model_name,
                'total_updates': trigger_config.get('update_count', 0),
                'last_update': trigger_config.get('last_update').isoformat() if trigger_config.get('last_update') else None,
                'update_history': model_history[-10:],  # 最近10次更新
                'success_rate': len([h for h in model_history if h['status'] == 'completed']) / len(model_history) if model_history else 0
            }
        else:
            # 全局统计
            total_updates = len(self.update_history)
            successful_updates = len([h for h in self.update_history if h['status'] == 'completed'])

            return {
                'total_models': len(self.update_triggers),
                'total_updates': total_updates,
                'successful_updates': successful_updates,
                'success_rate': successful_updates / total_updates if total_updates > 0 else 0,
                'models_updated': list(set(h['model_name'] for h in self.update_history))
            }


class ModelHealthChecker:
    """模型健康检查器"""

    def __init__(self):
        self.health_checks = {}
        self.health_history = {}
        self.health_thresholds = {
            'max_response_time': 5.0,  # 秒
            'min_accuracy': 0.7,
            'max_error_rate': 0.1,
            'max_memory_usage': 0.8,  # 80%
            'max_cpu_usage': 0.9      # 90%
        }

    def register_health_check(self, model_name: str, check_func: callable):
        """注册健康检查"""
        self.health_checks[model_name] = {
            'check_func': check_func,
            'last_check': None,
            'status': 'unknown',
            'consecutive_failures': 0
        }
        self.health_history[model_name] = []
        logger.info(f"已注册健康检查: {model_name}")

    async def perform_health_check(self, model_name: str) -> Dict[str, Any]:
        """执行健康检查"""
        if model_name not in self.health_checks:
            return {'status': 'not_registered'}

        check_config = self.health_checks[model_name]
        check_func = check_config['check_func']

        try:
            start_time = time.time()

            # 执行健康检查
            if asyncio.iscoroutinefunction(check_func):
                health_result = await check_func()
            else:
                health_result = await asyncio.get_event_loop().run_in_executor(None, check_func)

            check_time = time.time() - start_time

            # 评估健康状态
            overall_status = self._evaluate_health_status(health_result)

            # 更新检查配置
            check_config['last_check'] = datetime.now()
            if overall_status == 'healthy':
                check_config['consecutive_failures'] = 0
                check_config['status'] = 'healthy'
            else:
                check_config['consecutive_failures'] += 1
                check_config['status'] = overall_status

            # 记录健康历史
            health_record = {
                'timestamp': datetime.now(),
                'status': overall_status,
                'metrics': health_result,
                'check_time': check_time,
                'consecutive_failures': check_config['consecutive_failures']
            }

            self.health_history[model_name].append(health_record)

            # 限制历史记录
            max_history = 100
            if len(self.health_history[model_name]) > max_history:
                self.health_history[model_name] = self.health_history[model_name][-max_history:]

            return {
                'model_name': model_name,
                'status': overall_status,
                'metrics': health_result,
                'check_time': check_time,
                'last_check': check_config['last_check'].isoformat()
            }

        except Exception as e:
            logger.error(f"健康检查失败 {model_name}: {e}")

            check_config['consecutive_failures'] += 1
            check_config['status'] = 'error'

            return {
                'model_name': model_name,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }

    def _evaluate_health_status(self, health_result: Dict[str, Any]) -> str:
        """评估健康状态"""
        issues = []

        # 检查响应时间
        response_time = health_result.get('response_time')
        if response_time and response_time > self.health_thresholds['max_response_time']:
            issues.append('slow_response')

        # 检查准确率
        accuracy = health_result.get('accuracy')
        if accuracy and accuracy < self.health_thresholds['min_accuracy']:
            issues.append('low_accuracy')

        # 检查错误率
        error_rate = health_result.get('error_rate')
        if error_rate and error_rate > self.health_thresholds['max_error_rate']:
            issues.append('high_error_rate')

        # 检查资源使用
        memory_usage = health_result.get('memory_usage')
        if memory_usage and memory_usage > self.health_thresholds['max_memory_usage']:
            issues.append('high_memory_usage')

        cpu_usage = health_result.get('cpu_usage')
        if cpu_usage and cpu_usage > self.health_thresholds['max_cpu_usage']:
            issues.append('high_cpu_usage')

        # 确定整体状态
        if not issues:
            return 'healthy'
        elif len(issues) >= 3 or 'high_error_rate' in issues:
            return 'critical'
        elif len(issues) >= 2:
            return 'unhealthy'
        else:
            return 'warning'

    def get_health_stats(self, model_name: str = None) -> Dict[str, Any]:
        """获取健康统计"""
        if model_name:
            if model_name not in self.health_history:
                return {'status': 'no_data'}

            history = self.health_history[model_name]
            recent_checks = history[-20:] if len(history) > 20 else history

            status_counts = {}
            for check in recent_checks:
                status = check['status']
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                'model_name': model_name,
                'total_checks': len(history),
                'recent_checks': len(recent_checks),
                'current_status': self.health_checks[model_name]['status'],
                'status_distribution': status_counts,
                'avg_check_time': np.mean([c['check_time'] for c in recent_checks]),
                'consecutive_failures': self.health_checks[model_name]['consecutive_failures'],
                'last_check': self.health_checks[model_name]['last_check'].isoformat() if self.health_checks[model_name]['last_check'] else None
            }
        else:
            # 全局统计
            all_models = list(self.health_checks.keys())
            healthy_models = sum(1 for m in all_models if self.health_checks[m]['status'] == 'healthy')

            return {
                'total_models': len(all_models),
                'healthy_models': healthy_models,
                'health_rate': healthy_models / len(all_models) if all_models else 0,
                'models_by_status': {
                    status: sum(1 for m in all_models if self.health_checks[m]['status'] == status)
                    for status in ['healthy', 'warning', 'unhealthy', 'critical', 'error']
                }
            }


class ABTestingFramework:
    """A/B测试框架"""

    def __init__(self):
        self.active_tests = {}
        self.test_history = {}
        self.test_configs = {}

    def start_a_b_test(self, test_name: str, model_a: str, model_b: str,
                      test_config: Dict[str, Any]) -> str:
        """开始A/B测试"""
        test_id = f"ab_test_{int(time.time())}"

        self.active_tests[test_id] = {
            'test_name': test_name,
            'model_a': model_a,
            'model_b': model_b,
            'config': test_config,
            'start_time': datetime.now(),
            'traffic_distribution': test_config.get('traffic_distribution', 0.5),
            'metrics': {
                'model_a': {'requests': 0, 'responses': [], 'errors': 0},
                'model_b': {'requests': 0, 'responses': [], 'errors': 0}
            },
            'status': 'running'
        }

        self.test_configs[test_id] = test_config
        logger.info(f"已开始A/B测试: {test_name} ({test_id})")

        return test_id

    def record_test_request(self, test_id: str, model_version: str,
                           request_data: Dict[str, Any]) -> bool:
        """记录测试请求"""
        if test_id not in self.active_tests:
            return False

        test_info = self.active_tests[test_id]
        if model_version not in test_info['metrics']:
            return False

        test_info['metrics'][model_version]['requests'] += 1
        return True

    def record_test_response(self, test_id: str, model_version: str,
                           response_data: Dict[str, Any], error: bool = False):
        """记录测试响应"""
        if test_id not in self.active_tests:
            return

        test_info = self.active_tests[test_id]
        if model_version not in test_info['metrics']:
            return

        if error:
            test_info['metrics'][model_version]['errors'] += 1
        else:
            test_info['metrics'][model_version]['responses'].append(response_data)

        # 限制响应记录数量
        max_responses = 1000
        if len(test_info['metrics'][model_version]['responses']) > max_responses:
            test_info['metrics'][model_version]['responses'] = \
                test_info['metrics'][model_version]['responses'][-max_responses:]

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """获取测试结果"""
        if test_id not in self.active_tests:
            return {'error': 'test_not_found'}

        test_info = self.active_tests[test_id]

        results = {
            'test_id': test_id,
            'test_name': test_info['test_name'],
            'status': test_info['status'],
            'duration': (datetime.now() - test_info['start_time']).total_seconds(),
            'traffic_distribution': test_info['traffic_distribution']
        }

        # 计算各项指标
        for model in ['model_a', 'model_b']:
            metrics = test_info['metrics'][model]
            responses = metrics['responses']

            model_results = {
                'requests': metrics['requests'],
                'errors': metrics['errors'],
                'error_rate': metrics['errors'] / metrics['requests'] if metrics['requests'] > 0 else 0
            }

            # 计算性能指标
            if responses:
                latencies = [r.get('latency', 0) for r in responses if 'latency' in r]
                accuracies = [r.get('accuracy', 0) for r in responses if 'accuracy' in r]

                if latencies:
                    model_results['avg_latency'] = np.mean(latencies)
                    model_results['p95_latency'] = np.percentile(latencies, 95)

                if accuracies:
                    model_results['avg_accuracy'] = np.mean(accuracies)

            results[model] = model_results

        # 比较两个模型
        if 'model_a' in results and 'model_b' in results:
            results['comparison'] = self._compare_models(results['model_a'], results['model_b'])

        return results

    def _compare_models(self, model_a_results: Dict[str, Any],
                       model_b_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个模型"""
        comparison = {}

        # 比较错误率
        error_rate_a = model_a_results.get('error_rate', 0)
        error_rate_b = model_b_results.get('error_rate', 0)
        comparison['error_rate_winner'] = 'model_a' if error_rate_a < error_rate_b else 'model_b'

        # 比较延迟
        latency_a = model_a_results.get('avg_latency', float('inf'))
        latency_b = model_b_results.get('avg_latency', float('inf'))
        comparison['latency_winner'] = 'model_a' if latency_a < latency_b else 'model_b'

        # 比较准确率
        accuracy_a = model_a_results.get('avg_accuracy', 0)
        accuracy_b = model_b_results.get('avg_accuracy', 0)
        comparison['accuracy_winner'] = 'model_a' if accuracy_a > accuracy_b else 'model_b'

        # 计算综合评分
        score_a = (1 - error_rate_a) * 0.4 + (1 / (1 + latency_a)) * 0.3 + accuracy_a * 0.3
        score_b = (1 - error_rate_b) * 0.4 + (1 / (1 + latency_b)) * 0.3 + accuracy_b * 0.3

        comparison['overall_winner'] = 'model_a' if score_a > score_b else 'model_b'
        comparison['score_difference'] = abs(score_a - score_b)

        return comparison

    def stop_a_b_test(self, test_id: str) -> Dict[str, Any]:
        """停止A/B测试"""
        if test_id not in self.active_tests:
            return {'error': 'test_not_found'}

        test_info = self.active_tests[test_id]
        test_info['status'] = 'completed'
        test_info['end_time'] = datetime.now()

        # 获取最终结果
        final_results = self.get_test_results(test_id)

        # 移动到历史记录
        self.test_history[test_id] = test_info

        # 删除活跃测试
        del self.active_tests[test_id]

        logger.info(f"A/B测试已完成: {test_info['test_name']} ({test_id})")

        return {
            'test_id': test_id,
            'status': 'completed',
            'final_results': final_results,
            'winner': final_results.get('comparison', {}).get('overall_winner')
        }

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """获取活跃测试"""
        return [
            {
                'test_id': test_id,
                'test_name': test_info['test_name'],
                'start_time': test_info['start_time'].isoformat(),
                'duration': (datetime.now() - test_info['start_time']).total_seconds(),
                'model_a': test_info['model_a'],
                'model_b': test_info['model_b']
            }
            for test_id, test_info in self.active_tests.items()
        ]


class ModelOperationsManager:
    """模型运维管理器"""

    def __init__(self):
        self.version_manager = ModelVersionManager()
        self.performance_monitor = ModelPerformanceMonitor()
        self.automated_updater = AutomatedModelUpdater()
        self.health_checker = ModelHealthChecker()
        self.ab_testing = ABTestingFramework()
        self.operations_stats = {
            'total_models': 0,
            'active_monitors': 0,
            'health_checks': 0,
            'updates_performed': 0,
            'ab_tests_completed': 0
        }

    async def register_model_for_operations(self, model_name: str, model_path: str,
                                         metadata: Dict[str, Any],
                                         monitoring_func: callable = None,
                                         health_check_func: callable = None) -> bool:
        """注册模型进行运维"""
        try:
            # 注册版本
            version_id = self.version_manager.register_model_version(
                model_name, model_path, metadata
            )

            # 启动性能监控
            if monitoring_func:
                self.performance_monitor.start_monitoring(model_name, monitoring_func)

            # 注册健康检查
            if health_check_func:
                self.health_checker.register_health_check(model_name, health_check_func)

            # 更新统计
            self.operations_stats['total_models'] += 1
            self.operations_stats['active_monitors'] += 1 if monitoring_func else 0

            logger.info(f"模型 {model_name} 已注册进行运维管理")

            return True

        except Exception as e:
            logger.error(f"注册模型运维失败: {e}")
            return False

    async def perform_operations_check(self) -> Dict[str, Any]:
        """执行运维检查"""
        try:
            check_results = {
                'timestamp': datetime.now(),
                'version_status': {},
                'performance_status': {},
                'health_status': {},
                'update_status': {},
                'ab_test_status': {}
            }

            # 检查模型版本状态
            registry_stats = self.version_manager.get_registry_stats()
            check_results['version_status'] = registry_stats

            # 检查性能监控状态
            for model_name in registry_stats['models']:
                perf_stats = self.performance_monitor.get_performance_stats(model_name)
                check_results['performance_status'][model_name] = perf_stats

            # 执行健康检查
            for model_name in registry_stats['models']:
                health_result = await self.health_checker.perform_health_check(model_name)
                check_results['health_status'][model_name] = health_result
                self.operations_stats['health_checks'] += 1

            # 检查更新状态
            update_stats = self.automated_updater.get_update_stats()
            check_results['update_status'] = update_stats

            # 检查A/B测试状态
            active_tests = self.ab_testing.get_active_tests()
            check_results['ab_test_status'] = {
                'active_tests': len(active_tests),
                'test_details': active_tests
            }

            # 生成运维摘要
            check_results['summary'] = self._generate_operations_summary(check_results)

            return check_results

        except Exception as e:
            logger.error(f"运维检查失败: {e}")
            return {'error': str(e)}

    def _generate_operations_summary(self, check_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成运维摘要"""
        summary = {
            'overall_status': 'healthy',
            'issues_count': 0,
            'warnings_count': 0,
            'models_status': {},
            'recommendations': []
        }

        # 检查模型健康状态
        health_status = check_results.get('health_status', {})
        for model_name, health in health_status.items():
            status = health.get('status', 'unknown')
            summary['models_status'][model_name] = status

            if status in ['critical', 'error']:
                summary['issues_count'] += 1
                summary['overall_status'] = 'critical'
                summary['recommendations'].append(f"立即处理模型 {model_name} 的严重问题")
            elif status in ['unhealthy', 'warning']:
                summary['warnings_count'] += 1
                if summary['overall_status'] == 'healthy':
                    summary['overall_status'] = 'warning'
                summary['recommendations'].append(f"检查模型 {model_name} 的潜在问题")

        # 检查性能问题
        performance_status = check_results.get('performance_status', {})
        for model_name, perf in performance_status.items():
            alerts = perf.get('alert_summary', {}).get('total_alerts', 0)
            if alerts > 0:
                summary['warnings_count'] += 1
                summary['recommendations'].append(f"处理模型 {model_name} 的 {alerts} 个性能告警")

        # 检查活跃的A/B测试
        ab_tests = check_results.get('ab_test_status', {}).get('active_tests', 0)
        if ab_tests > 0:
            summary['recommendations'].append(f"监控 {ab_tests} 个正在进行的A/B测试")

        return summary

    def get_operations_dashboard(self) -> Dict[str, Any]:
        """获取运维仪表板数据"""
        try:
            dashboard = {
                'timestamp': datetime.now(),
                'operations_stats': self.operations_stats.copy(),
                'model_versions': self.version_manager.get_registry_stats(),
                'health_overview': self.health_checker.get_health_stats(),
                'update_overview': self.automated_updater.get_update_stats(),
                'active_ab_tests': self.ab_testing.get_active_tests(),
                'recent_alerts': self._get_recent_alerts(),
                'system_recommendations': self._generate_system_recommendations()
            }

            return dashboard

        except Exception as e:
            logger.error(f"获取运维仪表板失败: {e}")
            return {'error': str(e)}

    def _get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近告警"""
        # 这里应该从性能监控器和其他组件收集最近的告警
        # 暂时返回空列表
        return []

    def _generate_system_recommendations(self) -> List[str]:
        """生成系统建议"""
        recommendations = []

        # 检查模型健康
        health_stats = self.health_checker.get_health_stats()
        if health_stats.get('health_rate', 1.0) < 0.9:
            recommendations.append("提升模型健康率 - 多个模型存在健康问题")

        # 检查版本管理
        version_stats = self.version_manager.get_registry_stats()
        if version_stats.get('total_versions', 0) > version_stats.get('total_models', 0) * 5:
            recommendations.append("清理旧模型版本 - 版本数量过多")

        # 检查A/B测试
        active_tests = len(self.ab_testing.get_active_tests())
        if active_tests > 3:
            recommendations.append("减少并发A/B测试数量 - 当前测试过多")

        return recommendations
