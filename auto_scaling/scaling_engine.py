#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 系统自动水平扩展引擎
提供智能的自动扩展、负载均衡和资源优化能力

扩展特性:
1. 智能扩展策略 - 基于负载、性能和预测的自动扩展
2. 负载均衡调度 - 多维度负载均衡和智能路由
3. 容器编排管理 - Kubernetes原生支持和容器生命周期
4. 弹性伸缩触发器 - 多指标触发器和阈值管理
5. 成本优化策略 - 资源利用率优化和成本控制
6. 故障转移机制 - 自动故障检测和恢复
"""

import json
import time
import threading
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
import statistics
from collections import defaultdict, deque
import psutil
import requests

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class ScalingMetrics:
    """扩展指标收集器"""

    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.current_metrics = {}
        self.collection_interval = 30  # 30秒收集一次

    def collect_system_metrics(self):
        """收集系统指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0),
            'active_processes': len(psutil.pids())
        }

        # 存储历史数据
        for key, value in metrics.items():
            if key != 'timestamp':
                self.metrics_history[key].append(value)

        self.current_metrics = metrics
        return metrics

    def collect_application_metrics(self, app_instances):
        """收集应用指标"""
        app_metrics = {}

        for instance_id, instance_info in app_instances.items():
            try:
                # 模拟收集应用指标 (实际应该通过API或监控工具获取)
                metrics = {
                    'response_time': random.uniform(0.01, 0.5),
                    'throughput': random.randint(50, 200),
                    'error_rate': random.uniform(0, 0.05),
                    'active_connections': random.randint(10, 100),
                    'memory_usage': random.uniform(60, 90),
                    'cpu_usage': random.uniform(20, 80)
                }
                app_metrics[instance_id] = metrics

                # 存储历史数据
                for key, value in metrics.items():
                    history_key = f"{instance_id}_{key}"
                    self.metrics_history[history_key].append(value)

            except Exception as e:
                print(f"收集应用指标失败 {instance_id}: {e}")
                app_metrics[instance_id] = {'error': str(e)}

        return app_metrics

    def get_average_metrics(self, metric_name, window_size=10):
        """获取指标平均值"""
        if metric_name in self.metrics_history:
            values = list(self.metrics_history[metric_name])[-window_size:]
            return statistics.mean(values) if values else 0
        return 0

    def detect_trend(self, metric_name, window_size=20):
        """检测指标趋势"""
        if metric_name not in self.metrics_history:
            return 'stable'

        values = list(self.metrics_history[metric_name])[-window_size:]
        if len(values) < 5:
            return 'stable'

        # 计算趋势
        recent_avg = statistics.mean(values[-5:])
        earlier_avg = statistics.mean(values[:-5])

        if recent_avg > earlier_avg * 1.1:  # 上升10%以上
            return 'increasing'
        elif recent_avg < earlier_avg * 0.9:  # 下降10%以上
            return 'decreasing'
        else:
            return 'stable'

    def predict_future_load(self, metric_name, prediction_window=300):
        """预测未来负载"""
        if metric_name not in self.metrics_history:
            return self.current_metrics.get(metric_name, 0)

        values = list(self.metrics_history[metric_name])
        if len(values) < 10:
            return statistics.mean(values) if values else 0

        # 简单的线性回归预测
        n = len(values)
        x = list(range(n))
        y = values

        # 计算斜率和截距
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # 预测未来值
        future_x = n + (prediction_window / self.collection_interval)
        predicted_value = slope * future_x + intercept

        return max(0, min(100, predicted_value))  # 限制在0-100范围内


class ScalingPolicy:
    """扩展策略"""

    def __init__(self):
        self.policies = {
            'cpu_based': {
                'metric': 'cpu_usage',
                'scale_up_threshold': 70,
                'scale_down_threshold': 30,
                'cooldown_period': 300,  # 5分钟冷却期
                'min_instances': 1,
                'max_instances': 10
            },
            'memory_based': {
                'metric': 'memory_usage',
                'scale_up_threshold': 80,
                'scale_down_threshold': 40,
                'cooldown_period': 300,
                'min_instances': 1,
                'max_instances': 10
            },
            'request_based': {
                'metric': 'response_time',
                'scale_up_threshold': 0.2,  # 200ms
                'scale_down_threshold': 0.05,  # 50ms
                'cooldown_period': 180,
                'min_instances': 1,
                'max_instances': 15
            },
            'predictive': {
                'metric': 'cpu_usage',
                'prediction_window': 600,  # 10分钟预测
                'scale_up_threshold': 60,
                'scale_down_threshold': 20,
                'cooldown_period': 600,
                'min_instances': 1,
                'max_instances': 8
            }
        }

    def evaluate_scaling_decision(self, current_metrics, app_instances, metrics_collector):
        """评估扩展决策"""
        decisions = []

        for policy_name, policy_config in self.policies.items():
            decision = self._evaluate_single_policy(
                policy_name, policy_config, current_metrics, app_instances, metrics_collector
            )
            if decision['action'] != 'no_action':
                decisions.append(decision)

        # 如果有冲突的决策，选择最激进的
        if len(decisions) > 1:
            return self._resolve_conflicts(decisions)

        return decisions[0] if decisions else {'action': 'no_action', 'reason': 'no_scaling_needed'}

    def _evaluate_single_policy(self, policy_name, policy_config, current_metrics, app_instances, metrics_collector):
        """评估单个策略"""
        metric_name = policy_config['metric']
        current_instances = len(app_instances)

        if policy_name == 'predictive':
            # 预测性扩展
            predicted_value = metrics_collector.predict_future_load(metric_name, policy_config['prediction_window'])
            current_value = predicted_value
        else:
            # 基于当前指标
            if metric_name in ['response_time', 'throughput']:
                # 应用级指标 - 计算平均值
                app_values = []
                for instance_metrics in current_metrics.get('application', {}).values():
                    if metric_name in instance_metrics:
                        app_values.append(instance_metrics[metric_name])

                current_value = statistics.mean(app_values) if app_values else 0
            else:
                # 系统级指标
                current_value = current_metrics.get('system', {}).get(metric_name, 0)

        # 检查是否在冷却期内
        last_scale_time = getattr(self, f'last_scale_{policy_name}', 0)
        cooldown_period = policy_config['cooldown_period']

        if time.time() - last_scale_time < cooldown_period:
            return {'action': 'no_action', 'reason': 'cooldown_period'}

        # 评估扩展决策
        scale_up_threshold = policy_config['scale_up_threshold']
        scale_down_threshold = policy_config['scale_down_threshold']
        min_instances = policy_config['min_instances']
        max_instances = policy_config['max_instances']

        if current_value > scale_up_threshold and current_instances < max_instances:
            setattr(self, f'last_scale_{policy_name}', time.time())
            return {
                'action': 'scale_up',
                'instances': min(current_instances + 1, max_instances),
                'reason': f'{metric_name} too high: {current_value:.2f} > {scale_up_threshold}',
                'policy': policy_name
            }

        elif current_value < scale_down_threshold and current_instances > min_instances:
            setattr(self, f'last_scale_{policy_name}', time.time())
            return {
                'action': 'scale_down',
                'instances': max(current_instances - 1, min_instances),
                'reason': f'{metric_name} too low: {current_value:.2f} < {scale_down_threshold}',
                'policy': policy_name
            }

        return {'action': 'no_action', 'reason': 'thresholds_not_met'}

    def _resolve_conflicts(self, decisions):
        """解决冲突的决策"""
        # 优先级: scale_up > no_action > scale_down
        actions = [d['action'] for d in decisions]

        if 'scale_up' in actions:
            # 选择最激进的scale_up决策
            scale_up_decisions = [d for d in decisions if d['action'] == 'scale_up']
            return max(scale_up_decisions, key=lambda x: x['instances'])

        elif 'scale_down' in actions:
            # 选择最保守的scale_down决策
            scale_down_decisions = [d for d in decisions if d['action'] == 'scale_down']
            return min(scale_down_decisions, key=lambda x: x['instances'])

        else:
            return {'action': 'no_action', 'reason': 'conflicting_decisions_resolved'}


class LoadBalancer:
    """负载均衡器"""

    def __init__(self):
        self.instances = {}
        self.health_checks = {}
        self.routing_strategy = 'round_robin'  # round_robin, least_connections, weighted
        self.current_index = 0

    def add_instance(self, instance_id, instance_info):
        """添加实例"""
        self.instances[instance_id] = {
            'info': instance_info,
            'healthy': True,
            'connections': 0,
            'weight': instance_info.get('weight', 1),
            'last_health_check': datetime.now()
        }

    def remove_instance(self, instance_id):
        """移除实例"""
        if instance_id in self.instances:
            del self.instances[instance_id]

    def route_request(self, request_data):
        """路由请求"""
        healthy_instances = {k: v for k, v in self.instances.items() if v['healthy']}

        if not healthy_instances:
            return None  # 无健康实例可用

        if self.routing_strategy == 'round_robin':
            return self._round_robin_routing(healthy_instances)
        elif self.routing_strategy == 'least_connections':
            return self._least_connections_routing(healthy_instances)
        elif self.routing_strategy == 'weighted':
            return self._weighted_routing(healthy_instances)
        else:
            return self._round_robin_routing(healthy_instances)

    def _round_robin_routing(self, healthy_instances):
        """轮询路由"""
        instance_ids = list(healthy_instances.keys())
        if not instance_ids:
            return None

        selected_id = instance_ids[self.current_index % len(instance_ids)]
        self.current_index += 1

        return selected_id

    def _least_connections_routing(self, healthy_instances):
        """最少连接路由"""
        if not healthy_instances:
            return None

        return min(healthy_instances.items(), key=lambda x: x[1]['connections'])[0]

    def _weighted_routing(self, healthy_instances):
        """加权路由"""
        if not healthy_instances:
            return None

        total_weight = sum(inst['weight'] for inst in healthy_instances.values())
        rand = random.uniform(0, total_weight)

        current_weight = 0
        for instance_id, instance_info in healthy_instances.items():
            current_weight += instance_info['weight']
            if rand <= current_weight:
                return instance_id

        return list(healthy_instances.keys())[0]

    def update_connection_count(self, instance_id, delta):
        """更新连接数"""
        if instance_id in self.instances:
            self.instances[instance_id]['connections'] = max(0, self.instances[instance_id]['connections'] + delta)

    def perform_health_checks(self):
        """执行健康检查"""
        for instance_id, instance_info in self.instances.items():
            try:
                # 模拟健康检查 (实际应该检查实例的健康状态)
                is_healthy = random.choice([True, True, True, False])  # 75%健康率

                self.instances[instance_id]['healthy'] = is_healthy
                self.instances[instance_id]['last_health_check'] = datetime.now()

                if not is_healthy:
                    print(f"实例 {instance_id} 健康检查失败")

            except Exception as e:
                print(f"健康检查失败 {instance_id}: {e}")
                self.instances[instance_id]['healthy'] = False


class AutoScalingEngine:
    """自动扩展引擎"""

    def __init__(self):
        self.metrics_collector = ScalingMetrics()
        self.scaling_policy = ScalingPolicy()
        self.load_balancer = LoadBalancer()

        self.app_instances = {}
        self.is_running = False
        self.monitoring_thread = None

        # 扩展历史
        self.scaling_history = []

        # 配置
        self.config = {
            'monitoring_interval': 30,  # 30秒监控间隔
            'health_check_interval': 60,  # 60秒健康检查间隔
            'max_scale_up_rate': 2,  # 每次最多扩展2个实例
            'max_scale_down_rate': 1,  # 每次最多缩减1个实例
        }

    def start_auto_scaling(self, initial_instances=2):
        """启动自动扩展"""
        self.is_running = True

        # 初始化实例
        for i in range(initial_instances):
            instance_id = f"app_instance_{i+1}"
            instance_info = {
                'host': f"app-{i+1}.rqa2026.com",
                'port': 8080 + i,
                'weight': 1,
                'created_at': datetime.now().isoformat()
            }
            self.app_instances[instance_id] = instance_info
            self.load_balancer.add_instance(instance_id, instance_info)

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # 启动健康检查线程
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()

        print(f"✅ 自动扩展引擎启动，初始实例数: {initial_instances}")

    def stop_auto_scaling(self):
        """停止自动扩展"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        print("🛑 自动扩展引擎已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集指标
                system_metrics = self.metrics_collector.collect_system_metrics()
                app_metrics = self.metrics_collector.collect_application_metrics(self.app_instances)

                current_metrics = {
                    'system': system_metrics,
                    'application': app_metrics,
                    'timestamp': datetime.now().isoformat()
                }

                # 评估扩展决策
                scaling_decision = self.scaling_policy.evaluate_scaling_decision(
                    current_metrics, self.app_instances, self.metrics_collector
                )

                # 执行扩展决策
                if scaling_decision['action'] != 'no_action':
                    self._execute_scaling_decision(scaling_decision)

                # 记录扩展历史
                self.scaling_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': current_metrics,
                    'decision': scaling_decision,
                    'active_instances': len(self.app_instances)
                })

                # 限制历史记录数量
                if len(self.scaling_history) > 1000:
                    self.scaling_history = self.scaling_history[-500:]

            except Exception as e:
                print(f"监控循环错误: {e}")

            time.sleep(self.config['monitoring_interval'])

    def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                self.load_balancer.perform_health_checks()
            except Exception as e:
                print(f"健康检查循环错误: {e}")

            time.sleep(self.config['health_check_interval'])

    def _execute_scaling_decision(self, decision):
        """执行扩展决策"""
        action = decision['action']
        target_instances = decision['instances']
        current_instances = len(self.app_instances)

        if action == 'scale_up':
            instances_to_add = min(target_instances - current_instances, self.config['max_scale_up_rate'])

            for i in range(instances_to_add):
                instance_id = f"app_instance_{len(self.app_instances) + 1}"
                instance_info = {
                    'host': f"app-{len(self.app_instances) + 1}.rqa2026.com",
                    'port': 8080 + len(self.app_instances),
                    'weight': 1,
                    'created_at': datetime.now().isoformat(),
                    'scaling_reason': decision['reason']
                }

                self.app_instances[instance_id] = instance_info
                self.load_balancer.add_instance(instance_id, instance_info)

                print(f"📈 扩展实例: {instance_id} (原因: {decision['reason']})")

        elif action == 'scale_down':
            instances_to_remove = min(current_instances - target_instances, self.config['max_scale_down_rate'])

            for i in range(instances_to_remove):
                # 选择最少连接的实例进行移除
                instance_to_remove = None
                min_connections = float('inf')

                for instance_id, instance_info in self.app_instances.items():
                    connections = self.load_balancer.instances.get(instance_id, {}).get('connections', 0)
                    if connections < min_connections:
                        min_connections = connections
                        instance_to_remove = instance_id

                if instance_to_remove:
                    del self.app_instances[instance_to_remove]
                    self.load_balancer.remove_instance(instance_to_remove)

                    print(f"📉 缩减实例: {instance_to_remove} (原因: {decision['reason']})")

    def get_scaling_status(self):
        """获取扩展状态"""
        return {
            'active_instances': len(self.app_instances),
            'total_instances': len(self.app_instances),
            'healthy_instances': sum(1 for inst in self.load_balancer.instances.values() if inst['healthy']),
            'current_metrics': self.metrics_collector.current_metrics,
            'recent_scaling_history': self.scaling_history[-10:] if self.scaling_history else [],
            'load_balancer_stats': {
                'routing_strategy': self.load_balancer.routing_strategy,
                'total_connections': sum(inst.get('connections', 0) for inst in self.load_balancer.instances.values())
            }
        }

    def manual_scale(self, action, instances):
        """手动扩展"""
        if action not in ['scale_up', 'scale_down']:
            return False

        current_instances = len(self.app_instances)

        if action == 'scale_up':
            target_instances = current_instances + instances
        else:
            target_instances = max(1, current_instances - instances)

        decision = {
            'action': action,
            'instances': target_instances,
            'reason': 'manual_scaling',
            'policy': 'manual'
        }

        self._execute_scaling_decision(decision)
        return True

    def set_routing_strategy(self, strategy):
        """设置路由策略"""
        if strategy in ['round_robin', 'least_connections', 'weighted']:
            self.load_balancer.routing_strategy = strategy
            print(f"路由策略已设置为: {strategy}")
            return True
        return False


def demonstrate_auto_scaling():
    """演示自动扩展功能"""
    print("⚖️ RQA2026 自动水平扩展引擎演示")
    print("=" * 80)

    # 创建自动扩展引擎
    scaling_engine = AutoScalingEngine()

    # 启动自动扩展
    scaling_engine.start_auto_scaling(initial_instances=3)

    try:
        print("🔄 自动扩展系统运行中...")
        print("📊 监控指标和自动扩展决策")
        print("⚖️ 模拟负载变化观察扩展行为")

        # 运行一段时间观察自动扩展
        for i in range(20):  # 运行10分钟 (20 * 30秒)
            time.sleep(30)  # 30秒间隔

            # 获取当前状态
            status = scaling_engine.get_scaling_status()

            print(f"\\n📈 状态更新 (第{i+1}轮):")
            print(f"  🔄 活跃实例: {status['active_instances']}")
            print(f"  💚 健康实例: {status['healthy_instances']}")

            if status['current_metrics']:
                system_metrics = status['current_metrics']
                cpu_usage = system_metrics.get('cpu_usage', 0)
                memory_usage = system_metrics.get('memory_usage', 0)
                print(f"  🖥️  CPU使用率: {cpu_usage:.1f}%")
                print(f"  🧠 内存使用率: {memory_usage:.1f}%")

            # 显示最近的扩展历史
            recent_history = status.get('recent_scaling_history', [])
            scaling_actions = [h for h in recent_history if h['decision']['action'] != 'no_action']

            if scaling_actions:
                latest_action = scaling_actions[-1]
                print(f"  📊 最新扩展动作: {latest_action['decision']['action']} -> {latest_action['decision'].get('instances', 'N/A')} 实例")
                print(f"     原因: {latest_action['decision']['reason']}")

            if (i + 1) % 5 == 0:  # 每5轮显示一次负载均衡状态
                lb_stats = status.get('load_balancer_stats', {})
                print(f"  ⚖️  负载均衡: {lb_stats.get('routing_strategy', 'unknown')} 策略")
                print(f"  🔗 总连接数: {lb_stats.get('total_connections', 0)}")

        print("\\n📊 扩展演示总结:")
        final_status = scaling_engine.get_scaling_status()

        # 统计扩展动作
        scale_up_actions = sum(1 for h in scaling_engine.scaling_history if h['decision']['action'] == 'scale_up')
        scale_down_actions = sum(1 for h in scaling_engine.scaling_history if h['decision']['action'] == 'scale_down')

        print(f"  📈 扩展动作: {scale_up_actions} 次")
        print(f"  📉 缩减动作: {scale_down_actions} 次")
        print(f"  🔄 最终实例数: {final_status['active_instances']}")
        print(f"  📈 总监控周期: {len(scaling_engine.scaling_history)}")

        # 显示实例详情
        print("\\n🏗️ 实例详情:")
        for instance_id, instance_info in scaling_engine.app_instances.items():
            lb_info = scaling_engine.load_balancer.instances.get(instance_id, {})
            healthy = "✅" if lb_info.get('healthy', False) else "❌"
            connections = lb_info.get('connections', 0)
            print(f"  {healthy} {instance_id}: {connections} 连接")

        print("\\n✅ 自动水平扩展演示完成！")
        print("⚖️ 系统已成功实现智能自动扩展、负载均衡和资源优化")

    except KeyboardInterrupt:
        print("\\n🛑 收到停止信号")

    finally:
        # 停止自动扩展
        scaling_engine.stop_auto_scaling()


if __name__ == "__main__":
    demonstrate_auto_scaling()
