#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 优雅降级机制

实现系统降级和恢复，提升业务连续性
"""

import time
import logging
import threading
import json
from typing import Dict, Any, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):

    """服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class CircuitBreakerState(Enum):

    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ServiceHealthChecker:

    """服务健康检查器"""

    def __init__(self):

        self.services = {}
        self.check_interval = 30  # 30秒检查一次
        self.failure_threshold = 3  # 失败阈值
        self.recovery_threshold = 2  # 恢复阈值

    def register_service(self, service_name: str, health_check_func: Callable):
        """注册服务"""
        self.services[service_name] = {
            'health_check': health_check_func,
            'status': ServiceStatus.HEALTHY,
            'failure_count': 0,
            'success_count': 0,
            'last_check': None,
            'consecutive_failures': 0
        }
        logger.info(f"注册服务: {service_name}")

    def check_service_health(self, service_name: str) -> ServiceStatus:
        """检查服务健康状态"""
        if service_name not in self.services:
            return ServiceStatus.DOWN

        service = self.services[service_name]
        health_check = service['health_check']

        try:
            # 执行健康检查
            is_healthy = health_check()
            service['last_check'] = time.time()

            if is_healthy:
                service['success_count'] += 1
                service['consecutive_failures'] = 0

                # 检查是否可以恢复
                if service['status'] != ServiceStatus.HEALTHY:
                    if service['success_count'] >= self.recovery_threshold:
                        service['status'] = ServiceStatus.HEALTHY
                        logger.info(f"服务恢复: {service_name}")
            else:
                service['failure_count'] += 1
                service['consecutive_failures'] += 1
                service['success_count'] = 0

                # 根据失败次数设置状态
                if service['consecutive_failures'] >= self.failure_threshold:
                    if service['consecutive_failures'] >= 10:
                        service['status'] = ServiceStatus.DOWN
                    elif service['consecutive_failures'] >= 5:
                        service['status'] = ServiceStatus.CRITICAL
                    else:
                        service['status'] = ServiceStatus.DEGRADED

        except Exception as e:
            logger.error(f"健康检查失败 {service_name}: {e}")
            service['status'] = ServiceStatus.DOWN

        return service['status']

    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """获取服务状态"""
        if service_name not in self.services:
            return {'status': ServiceStatus.DOWN}

        service = self.services[service_name]
        return {
            'status': service['status'],
            'failure_count': service['failure_count'],
            'success_count': service['success_count'],
            'consecutive_failures': service['consecutive_failures'],
            'last_check': service['last_check']
        }


class CircuitBreaker:

    """熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    def call(self, func: Callable, *args, **kwargs):
        """调用函数，带熔断保护"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

            raise e


class GracefulDegradationManager:

    """优雅降级管理器"""

    def __init__(self):

        self.health_checker = ServiceHealthChecker()
        self.circuit_breakers = {}
        self.degradation_strategies = {}
        self.fallback_functions = {}

    def register_service_with_degradation(self,


                                          service_name: str,
                                          primary_func: Callable,
                                          fallback_func: Callable,
                                          health_check_func: Callable,
                                          degradation_strategy: str = "circuit_breaker"):
        """注册带降级的服务"""
        # 注册健康检查
        self.health_checker.register_service(service_name, health_check_func)

        # 创建熔断器
        if degradation_strategy == "circuit_breaker":
            self.circuit_breakers[service_name] = CircuitBreaker()

        # 保存函数
        self.degradation_strategies[service_name] = {
            'primary': primary_func,
            'fallback': fallback_func,
            'strategy': degradation_strategy,
            'health_check': health_check_func
        }

        logger.info(f"注册降级服务: {service_name}")

    def call_with_degradation(self, service_name: str, *args, **kwargs):
        """带降级的服务调用"""
        if service_name not in self.degradation_strategies:
            raise Exception(f"Service not registered: {service_name}")

        strategy = self.degradation_strategies[service_name]
        service_status = self.health_checker.check_service_health(service_name)

        try:
            if service_status == ServiceStatus.HEALTHY:
                # 使用熔断器调用主要服务
                if service_name in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[service_name]
                    return circuit_breaker.call(strategy['primary'], *args, **kwargs)
                else:
                    return strategy['primary'](*args, **kwargs)
            else:
                # 服务不健康，使用降级方案
                logger.warning(f"服务 {service_name} 状态: {service_status}, 使用降级方案")
                return strategy['fallback'](*args, **kwargs)

        except Exception as e:
            logger.error(f"服务 {service_name} 调用失败: {e}")
            # 强制使用降级方案
            return strategy['fallback'](*args, **kwargs)

# 示例服务函数


def database_service_primary(query: str) -> Dict[str, Any]:
    """主要数据库服务"""
    # 模拟正常数据库操作
    time.sleep(0.1)  # 模拟查询时间
    return {
        'status': 'success',
        'data': f"Query result for: {query}",
        'source': 'primary_database',
        'timestamp': datetime.now().isoformat()
    }


def database_service_fallback(query: str) -> Dict[str, Any]:
    """降级数据库服务"""
    # 模拟降级方案：返回缓存数据或简化结果
    time.sleep(0.05)  # 降级方案更快
    return {
        'status': 'degraded',
        'data': f"Cached result for: {query}",
        'source': 'cache_fallback',
        'timestamp': datetime.now().isoformat()
    }


def check_database_health() -> bool:
    """检查数据库健康状态"""
    # 模拟健康检查
    import secrets
    return secrets.random() > 0.1  # 90 % 正常


def ai_service_primary(prompt: str) -> Dict[str, Any]:
    """主要AI服务"""
    time.sleep(0.5)  # 模拟AI推理时间
    return {
        'status': 'success',
        'response': f"AI response to: {prompt}",
        'model': 'advanced_model',
        'timestamp': datetime.now().isoformat()
    }


def ai_service_fallback(prompt: str) -> Dict[str, Any]:
    """降级AI服务"""
    time.sleep(0.1)  # 降级方案更快
    return {
        'status': 'degraded',
        'response': f"Simple response to: {prompt}",
        'model': 'basic_model',
        'timestamp': datetime.now().isoformat()
    }


def check_ai_service_health() -> bool:
    """检查AI服务健康状态"""
    import secrets
    return secrets.random() > 0.2  # 80 % 正常


def test_graceful_degradation():
    """测试优雅降级机制"""
    print("测试优雅降级机制...")

    # 创建降级管理器
    manager = GracefulDegradationManager()

    # 注册数据库服务
    manager.register_service_with_degradation(
        'database',
        database_service_primary,
        database_service_fallback,
        check_database_health
    )

    # 注册AI服务
    manager.register_service_with_degradation(
        'ai_service',
        ai_service_primary,
        ai_service_fallback,
        check_ai_service_health
    )

    # 测试多次调用
    test_results = []
    for i in range(10):
        print(f"\n第 {i + 1} 轮测试:")

        # 测试数据库服务
        try:
            db_result = manager.call_with_degradation(
                'database', f"SELECT * FROM users WHERE id = {i}")
            print(f"  数据库服务: {db_result['status']} ({db_result['source']})")
        except Exception as e:
            print(f"  数据库服务异常: {e}")

        # 测试AI服务
        try:
            ai_result = manager.call_with_degradation('ai_service', f"请分析用户{i}的行为模式")
            print(f"  AI服务: {ai_result['status']} ({ai_result['model']})")
        except Exception as e:
            print(f"  AI服务异常: {e}")

        # 检查服务状态
        db_status = manager.health_checker.get_service_status('database')
        ai_status = manager.health_checker.get_service_status('ai_service')

        print(f"  数据库状态: {db_status['status'].value} (失败: {db_status['consecutive_failures']})")
        print(f"  AI服务状态: {ai_status['status'].value} (失败: {ai_status['consecutive_failures']})")

        test_results.append({
            'round': i + 1,
            'db_status': db_status['status'].value,
            'ai_status': ai_status['status'].value
        })

        time.sleep(0.5)  # 短暂延迟

    # 统计结果
    db_degraded = sum(1 for r in test_results if r['db_status'] != 'healthy')
    ai_degraded = sum(1 for r in test_results if r['ai_status'] != 'healthy')

    print("\n测试统计:")
    print(f"  数据库降级次数: {db_degraded}/10")
    print(f"  AI服务降级次数: {ai_degraded}/10")
    print(".1f")
    print(".1f")
    return {
        'total_tests': 10,
        'db_degraded': db_degraded,
        'ai_degraded': ai_degraded,
        'db_degradation_rate': db_degraded / 10 * 100,
        'ai_degradation_rate': ai_degraded / 10 * 100
    }


if __name__ == "__main__":
    print("优雅降级机制测试...")

    # 运行测试
    stats = test_graceful_degradation()

    print("\n✅ 优雅降级测试完成")
    print(
        f"📊 测试结果: 数据库降级率 {stats['db_degradation_rate']:.1f}%, AI服务降级率 {stats['ai_degradation_rate']:.1f}%")

    # 保存测试结果
    test_results = {
        'graceful_degradation_stats': stats,
        'timestamp': datetime.now().isoformat(),
        'test_description': '优雅降级机制测试'
    }

    with open('graceful_degradation_test_results.json', 'w', encoding='utf - 8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)

    print("📁 测试结果已保存: graceful_degradation_test_results.json")


class AdaptiveHealthChecker(ServiceHealthChecker):

    """自适应健康检查器 - 动态调整检查频率"""

    def __init__(self):

        super().__init__()
        # 自适应检查间隔配置
        self.min_interval = 10      # 最小检查间隔（秒）
        self.max_interval = 300     # 最大检查间隔（秒）
        self.base_interval = 30     # 基础检查间隔（秒）

        # 动态调整参数
        self.failure_penalty = 1.5  # 失败时的检查频率惩罚倍数
        self.success_bonus = 0.8    # 成功时的检查频率奖励倍数
        self.stability_threshold = 10  # 稳定性阈值（连续成功次数）

        # 服务特定的自适应状态
        self.adaptive_states = {}

        # 性能统计
        self.stats = {
            'total_checks': 0,
            'adaptive_adjustments': 0,
            'average_interval': 0.0,
            'interval_changes': []
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def register_service(self, service_name: str, health_check_func: Callable):
        """注册服务并初始化自适应状态"""
        super().register_service(service_name, health_check_func)

        # 初始化自适应状态
        self.adaptive_states[service_name] = {
            'current_interval': self.base_interval,
            'last_interval_change': time.time(),
            'interval_history': [],
            'stability_score': 0.0,
            'trend_direction': 'stable'  # stable, increasing, decreasing
        }

        self.logger.info(f"注册自适应健康检查服务: {service_name}")

    def check_service_health(self, service_name: str) -> ServiceStatus:
        """执行自适应健康检查"""
        if service_name not in self.services:
            return ServiceStatus.DOWN

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        # 记录检查开始时间
        check_start_time = time.time()

        # 执行健康检查
        status = super().check_service_health(service_name)

        # 更新自适应状态
        self._update_adaptive_state(service_name, status)

        # 计算下次检查间隔
        next_interval = self.get_adaptive_interval(service_name)

        # 记录统计信息
        check_duration = time.time() - check_start_time
        self.stats['total_checks'] += 1

        # 更新平均间隔
        total_checks = self.stats['total_checks']
        current_avg = self.stats['average_interval']
        self.stats['average_interval'] = (
            (current_avg * (total_checks - 1)) + next_interval
        ) / total_checks

        # 记录间隔变化
        if len(adaptive_state['interval_history']) >= 2:
            prev_interval = adaptive_state['interval_history'][-2]
            if abs(next_interval - prev_interval) > 1.0:  # 间隔变化超过1秒
                self.stats['adaptive_adjustments'] += 1
                self.stats['interval_changes'].append({
                    'service': service_name,
                    'timestamp': time.time(),
                    'from_interval': prev_interval,
                    'to_interval': next_interval,
                    'reason': self._get_interval_change_reason(service_name)
                })

        self.logger.debug(
            f"自适应健康检查 {service_name}: 状态={status.value}, "
            f"下次间隔={next_interval:.1f}s, 检查耗时={check_duration:.3f}s"
        )

        return status

    def get_adaptive_interval(self, service_name: str) -> float:
        """
        获取自适应检查间隔

        Args:
            service_name: 服务名称

        Returns:
            float: 自适应检查间隔（秒）
        """
        if service_name not in self.adaptive_states:
            return self.base_interval

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        # 基于连续失败次数调整间隔
        if consecutive_failures > 0:
            # 失败时增加检查频率（减少间隔）
            adaptive_interval = self.base_interval * (self.failure_penalty ** consecutive_failures)
        elif success_count > self.stability_threshold:
            # 成功稳定时减少检查频率（增加间隔）
            adaptive_interval = self.base_interval * self.success_bonus
        else:
            adaptive_interval = self.base_interval

        # 限制在合理范围内
        adaptive_interval = min(max(adaptive_interval, self.min_interval), self.max_interval)

        # 更新当前间隔
        adaptive_state['current_interval'] = adaptive_interval
        adaptive_state['interval_history'].append(adaptive_interval)

        # 保持历史记录在合理长度
        if len(adaptive_state['interval_history']) > 100:
            adaptive_state['interval_history'] = adaptive_state['interval_history'][-50:]

        return adaptive_interval

    def _update_adaptive_state(self, service_name: str, status: ServiceStatus):
        """更新自适应状态"""
        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        # 计算稳定性分数
        if consecutive_failures == 0 and success_count > 0:
            # 成功状态，增加稳定性
            adaptive_state['stability_score'] = min(
                adaptive_state['stability_score'] + 0.1,
                1.0
            )
        elif consecutive_failures > 0:
            # 失败状态，降低稳定性
            adaptive_state['stability_score'] = max(
                adaptive_state['stability_score'] - 0.2,
                0.0
            )

        # 更新趋势方向
        if len(adaptive_state['interval_history']) >= 3:
            recent_intervals = adaptive_state['interval_history'][-3:]
            if recent_intervals[-1] > recent_intervals[-2] > recent_intervals[-3]:
                adaptive_state['trend_direction'] = 'increasing'
            elif recent_intervals[-1] < recent_intervals[-2] < recent_intervals[-3]:
                adaptive_state['trend_direction'] = 'decreasing'
            else:
                adaptive_state['trend_direction'] = 'stable'

    def _get_interval_change_reason(self, service_name: str) -> str:
        """获取间隔变化的原因"""
        service = self.services[service_name]
        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        if consecutive_failures > 0:
            return f"连续失败 {consecutive_failures} 次"
        elif success_count > self.stability_threshold:
            return f"连续成功 {success_count} 次，服务稳定"
        else:
            return "正常调整"

    def get_service_adaptive_info(self, service_name: str) -> Dict[str, Any]:
        """获取服务的自适应信息"""
        if service_name not in self.services or service_name not in self.adaptive_states:
            return {}

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        return {
            'service_name': service_name,
            'current_interval': adaptive_state['current_interval'],
            'stability_score': adaptive_state['stability_score'],
            'trend_direction': adaptive_state['trend_direction'],
            'consecutive_failures': service.get('consecutive_failures', 0),
            'success_count': service.get('success_count', 0),
            'interval_history_length': len(adaptive_state['interval_history']),
            'next_check_interval': self.get_adaptive_interval(service_name)
        }

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """获取自适应检查统计信息"""
        stats = self.stats.copy()
        stats['services_info'] = {}

        for service_name in self.services.keys():
            stats['services_info'][service_name] = self.get_service_adaptive_info(service_name)

        return stats

    def reset_service_adaptive_state(self, service_name: str):
        """重置服务的自适应状态"""
        if service_name in self.adaptive_states:
            self.adaptive_states[service_name] = {
                'current_interval': self.base_interval,
                'last_interval_change': time.time(),
                'interval_history': [self.base_interval],
                'stability_score': 0.0,
                'trend_direction': 'stable'
            }
            self.logger.info(f"重置服务 {service_name} 的自适应状态")

    def set_adaptive_parameters(self, min_interval: float = None, max_interval: float = None,


                                failure_penalty: float = None, success_bonus: float = None):
        """设置自适应参数"""
        if min_interval is not None:
            self.min_interval = max(min_interval, 1.0)  # 最少1秒
        if max_interval is not None:
            self.max_interval = min(max_interval, 3600.0)  # 最多1小时
        if failure_penalty is not None:
            self.failure_penalty = max(failure_penalty, 1.0)
        if success_bonus is not None:
            self.success_bonus = max(success_bonus, 0.1)

        # 确保参数合理性
        if self.min_interval >= self.max_interval:
            self.max_interval = self.min_interval * 10

        self.logger.info(
            f"更新自适应参数: min={self.min_interval}, max={self.max_interval}, "
            f"penalty={self.failure_penalty}, bonus={self.success_bonus}"
        )


class AdaptiveHealthCheckScheduler:

    """自适应健康检查调度器"""

    def __init__(self, health_checker: AdaptiveHealthChecker):

        self.health_checker = health_checker
        self.scheduled_checks = {}
        self.running = False
        self.scheduler_thread = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_scheduler(self):
        """启动调度器"""
        if self.running:
            self.logger.warning("调度器已经在运行")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("自适应健康检查调度器已启动")

    def stop_scheduler(self):
        """停止调度器"""
        if not self.running:
            self.logger.warning("调度器未在运行")
            return

        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        self.logger.info("自适应健康检查调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                current_time = time.time()

                # 检查所有服务的下次检查时间
                for service_name in list(self.health_checker.services.keys()):
                    if service_name not in self.scheduled_checks:
                        # 新服务，立即安排检查
                        self.scheduled_checks[service_name] = current_time
                        continue

                    next_check_time = self.scheduled_checks[service_name]
                    if current_time >= next_check_time:
                        # 执行健康检查
                        status = self.health_checker.check_service_health(service_name)

                        # 计算下次检查时间
                        interval = self.health_checker.get_adaptive_interval(service_name)
                        self.scheduled_checks[service_name] = current_time + interval

                        self.logger.debug(
                            f"执行定时健康检查 {service_name}: 状态={status.value}, "
                            f"下次检查间隔={interval:.1f}s"
                        )

                # 短暂休眠，避免CPU占用过高
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                time.sleep(5.0)  # 异常时稍长休眠


def test_adaptive_health_checker():
    """测试自适应健康检查器"""
    print("🧪 测试自适应健康检查器...")

    # 创建自适应健康检查器
    checker = AdaptiveHealthChecker()

    # 模拟服务健康检查函数

    def mock_service_check():

        import secrets
        # 80% 成功率
        return secrets.random() > 0.2

    # 注册服务
    checker.register_service("test_service", mock_service_check)

    print("📊 初始状态:")
    info = checker.get_service_adaptive_info("test_service")
    print(f"  当前间隔: {info['current_interval']}s")
    print(f"  稳定性分数: {info['stability_score']}")
    print(f"  趋势方向: {info['trend_direction']}")

    # 执行多次检查
    print("\n🔄 执行健康检查...")
    results = []
    for i in range(15):
        status = checker.check_service_health("test_service")
        info = checker.get_service_adaptive_info("test_service")

        results.append({
            'check': i + 1,
            'status': status.value,
            'interval': info['current_interval'],
            'stability': info['stability_score'],
            'consecutive_failures': info['consecutive_failures']
        })

        print(f"  检查 {i + 1}: 状态={status.value}, 间隔={info['current_interval']:.1f}s, "
              f"稳定性={info['stability_score']:.2f}")

        time.sleep(0.1)  # 短暂延迟

    # 统计结果
    successful_checks = sum(1 for r in results if r['status'] == 'healthy')
    avg_interval = sum(r['interval'] for r in results) / len(results)
    final_stability = results[-1]['stability']

    print("\n📈 测试统计:")
    print(f"  总检查次数: {len(results)}")
    print(f"  成功检查数: {successful_checks}")
    print(".1f")
    print(".2f")
    print(".1f")
    # 获取自适应统计
    adaptive_stats = checker.get_adaptive_stats()
    print("\n🔧 自适应统计:")
    print(f"  总检查数: {adaptive_stats['total_checks']}")
    print(f"  自适应调整次数: {adaptive_stats['adaptive_adjustments']}")
    print(".1f")
    print(f"  间隔变化记录数: {len(adaptive_stats['interval_changes'])}")

    return {
        'total_checks': len(results),
        'successful_checks': successful_checks,
        'success_rate': successful_checks / len(results) * 100,
        'average_interval': avg_interval,
        'final_stability': final_stability,
        'adaptive_stats': adaptive_stats
    }


if __name__ == "__main__":
    print("自适应健康检查器测试...")

    # 运行测试
    test_results = test_adaptive_health_checker()

    print("\n✅ 自适应健康检查器测试完成")
    print(".1f")
    print(".1f")
    print(".2f")
    # 保存测试结果
    test_output = {
        'adaptive_health_check_test': test_results,
        'timestamp': datetime.now().isoformat(),
        'test_description': '自适应健康检查器功能测试'
    }

    with open('adaptive_health_check_test_results.json', 'w', encoding='utf - 8') as f:
        json.dump(test_output, f, ensure_ascii=False, indent=2, default=str)

    print("📁 测试结果已保存: adaptive_health_check_test_results.json")


class AdaptiveHealthChecker(ServiceHealthChecker):

    """自适应健康检查器 - 动态调整检查频率"""

    def __init__(self):

        super().__init__()
        # 自适应检查间隔配置
        self.min_interval = 10      # 最小检查间隔（秒）
        self.max_interval = 300     # 最大检查间隔（秒）
        self.base_interval = 30     # 基础检查间隔（秒）

        # 动态调整参数
        self.failure_penalty = 1.5  # 失败时的检查频率惩罚倍数
        self.success_bonus = 0.8    # 成功时的检查频率奖励倍数
        self.stability_threshold = 10  # 稳定性阈值（连续成功次数）

        # 服务特定的自适应状态
        self.adaptive_states = {}

        # 性能统计
        self.stats = {
            'total_checks': 0,
            'adaptive_adjustments': 0,
            'average_interval': 0.0,
            'interval_changes': []
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def register_service(self, service_name: str, health_check_func: Callable):
        """注册服务并初始化自适应状态"""
        super().register_service(service_name, health_check_func)

        # 初始化自适应状态
        self.adaptive_states[service_name] = {
            'current_interval': self.base_interval,
            'last_interval_change': time.time(),
            'interval_history': [],
            'stability_score': 0.0,
            'trend_direction': 'stable'  # stable, increasing, decreasing
        }

        self.logger.info(f"注册自适应健康检查服务: {service_name}")

    def check_service_health(self, service_name: str) -> ServiceStatus:
        """执行自适应健康检查"""
        if service_name not in self.services:
            return ServiceStatus.DOWN

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        # 记录检查开始时间
        check_start_time = time.time()

        # 执行健康检查
        status = super().check_service_health(service_name)

        # 更新自适应状态
        self._update_adaptive_state(service_name, status)

        # 计算下次检查间隔
        next_interval = self.get_adaptive_interval(service_name)

        # 记录统计信息
        check_duration = time.time() - check_start_time
        self.stats['total_checks'] += 1

        # 更新平均间隔
        total_checks = self.stats['total_checks']
        current_avg = self.stats['average_interval']
        self.stats['average_interval'] = (
            (current_avg * (total_checks - 1)) + next_interval
        ) / total_checks

        # 记录间隔变化
        if len(adaptive_state['interval_history']) >= 2:
            prev_interval = adaptive_state['interval_history'][-2]
            if abs(next_interval - prev_interval) > 1.0:  # 间隔变化超过1秒
                self.stats['adaptive_adjustments'] += 1
                self.stats['interval_changes'].append({
                    'service': service_name,
                    'timestamp': time.time(),
                    'from_interval': prev_interval,
                    'to_interval': next_interval,
                    'reason': self._get_interval_change_reason(service_name)
                })

        self.logger.debug(
            f"自适应健康检查 {service_name}: 状态={status.value}, "
            f"下次间隔={next_interval:.1f}s, 检查耗时={check_duration:.3f}s"
        )

        return status

    def get_adaptive_interval(self, service_name: str) -> float:
        """
        获取自适应检查间隔

        Args:
            service_name: 服务名称

        Returns:
            float: 自适应检查间隔（秒）
        """
        if service_name not in self.adaptive_states:
            return self.base_interval

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        # 基于连续失败次数调整间隔
        if consecutive_failures > 0:
            # 失败时增加检查频率（减少间隔）
            adaptive_interval = self.base_interval * (self.failure_penalty ** consecutive_failures)
        elif success_count > self.stability_threshold:
            # 成功稳定时减少检查频率（增加间隔）
            adaptive_interval = self.base_interval * self.success_bonus
        else:
            adaptive_interval = self.base_interval

        # 限制在合理范围内
        adaptive_interval = min(max(adaptive_interval, self.min_interval), self.max_interval)

        # 更新当前间隔
        adaptive_state['current_interval'] = adaptive_interval
        adaptive_state['interval_history'].append(adaptive_interval)

        # 保持历史记录在合理长度
        if len(adaptive_state['interval_history']) > 100:
            adaptive_state['interval_history'] = adaptive_state['interval_history'][-50:]

        return adaptive_interval

    def _update_adaptive_state(self, service_name: str, status: ServiceStatus):
        """更新自适应状态"""
        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        # 计算稳定性分数
        if consecutive_failures == 0 and success_count > 0:
            # 成功状态，增加稳定性
            adaptive_state['stability_score'] = min(
                adaptive_state['stability_score'] + 0.1,
                1.0
            )
        elif consecutive_failures > 0:
            # 失败状态，降低稳定性
            adaptive_state['stability_score'] = max(
                adaptive_state['stability_score'] - 0.2,
                0.0
            )

        # 更新趋势方向
        if len(adaptive_state['interval_history']) >= 3:
            recent_intervals = adaptive_state['interval_history'][-3:]
            if recent_intervals[-1] > recent_intervals[-2] > recent_intervals[-3]:
                adaptive_state['trend_direction'] = 'increasing'
            elif recent_intervals[-1] < recent_intervals[-2] < recent_intervals[-3]:
                adaptive_state['trend_direction'] = 'decreasing'
            else:
                adaptive_state['trend_direction'] = 'stable'

    def _get_interval_change_reason(self, service_name: str) -> str:
        """获取间隔变化的原因"""
        service = self.services[service_name]
        consecutive_failures = service.get('consecutive_failures', 0)
        success_count = service.get('success_count', 0)

        if consecutive_failures > 0:
            return f"连续失败 {consecutive_failures} 次"
        elif success_count > self.stability_threshold:
            return f"连续成功 {success_count} 次，服务稳定"
        else:
            return "正常调整"

    def get_service_adaptive_info(self, service_name: str) -> Dict[str, Any]:
        """获取服务的自适应信息"""
        if service_name not in self.services or service_name not in self.adaptive_states:
            return {}

        service = self.services[service_name]
        adaptive_state = self.adaptive_states[service_name]

        return {
            'service_name': service_name,
            'current_interval': adaptive_state['current_interval'],
            'stability_score': adaptive_state['stability_score'],
            'trend_direction': adaptive_state['trend_direction'],
            'consecutive_failures': service.get('consecutive_failures', 0),
            'success_count': service.get('success_count', 0),
            'interval_history_length': len(adaptive_state['interval_history']),
            'next_check_interval': self.get_adaptive_interval(service_name)
        }

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """获取自适应检查统计信息"""
        stats = self.stats.copy()
        stats['services_info'] = {}

        for service_name in self.services.keys():
            stats['services_info'][service_name] = self.get_service_adaptive_info(service_name)

        return stats

    def reset_service_adaptive_state(self, service_name: str):
        """重置服务的自适应状态"""
        if service_name in self.adaptive_states:
            self.adaptive_states[service_name] = {
                'current_interval': self.base_interval,
                'last_interval_change': time.time(),
                'interval_history': [self.base_interval],
                'stability_score': 0.0,
                'trend_direction': 'stable'
            }
            self.logger.info(f"重置服务 {service_name} 的自适应状态")

    def set_adaptive_parameters(self, min_interval: float = None, max_interval: float = None,


                                failure_penalty: float = None, success_bonus: float = None):
        """设置自适应参数"""
        if min_interval is not None:
            self.min_interval = max(min_interval, 1.0)  # 最少1秒
        if max_interval is not None:
            self.max_interval = min(max_interval, 3600.0)  # 最多1小时
        if failure_penalty is not None:
            self.failure_penalty = max(failure_penalty, 1.0)
        if success_bonus is not None:
            self.success_bonus = max(success_bonus, 0.1)

        # 确保参数合理性
        if self.min_interval >= self.max_interval:
            self.max_interval = self.min_interval * 10

        self.logger.info(
            f"更新自适应参数: min={self.min_interval}, max={self.max_interval}, "
            f"penalty={self.failure_penalty}, bonus={self.success_bonus}"
        )


class AdaptiveHealthCheckScheduler:

    """自适应健康检查调度器"""

    def __init__(self, health_checker: AdaptiveHealthChecker):

        self.health_checker = health_checker
        self.scheduled_checks = {}
        self.running = False
        self.scheduler_thread = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_scheduler(self):
        """启动调度器"""
        if self.running:
            self.logger.warning("调度器已经在运行")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("自适应健康检查调度器已启动")

    def stop_scheduler(self):
        """停止调度器"""
        if not self.running:
            self.logger.warning("调度器未在运行")
            return

        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        self.logger.info("自适应健康检查调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                current_time = time.time()

                # 检查所有服务的下次检查时间
                for service_name in list(self.health_checker.services.keys()):
                    if service_name not in self.scheduled_checks:
                        # 新服务，立即安排检查
                        self.scheduled_checks[service_name] = current_time
                        continue

                    next_check_time = self.scheduled_checks[service_name]
                    if current_time >= next_check_time:
                        # 执行健康检查
                        status = self.health_checker.check_service_health(service_name)

                        # 计算下次检查时间
                        interval = self.health_checker.get_adaptive_interval(service_name)
                        self.scheduled_checks[service_name] = current_time + interval

                        self.logger.debug(
                            f"执行定时健康检查 {service_name}: 状态={status.value}, "
                            f"下次检查间隔={interval:.1f}s"
                        )

                # 短暂休眠，避免CPU占用过高
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                time.sleep(5.0)  # 异常时稍长休眠


def test_adaptive_health_checker():
    """测试自适应健康检查器"""
    print("🧪 测试自适应健康检查器...")

    # 创建自适应健康检查器
    checker = AdaptiveHealthChecker()

    # 模拟服务健康检查函数

    def mock_service_check():

        import secrets
        # 80% 成功率
        return secrets.random() > 0.2

    # 注册服务
    checker.register_service("test_service", mock_service_check)

    print("📊 初始状态:")
    info = checker.get_service_adaptive_info("test_service")
    print(f"  当前间隔: {info['current_interval']}s")
    print(f"  稳定性分数: {info['stability_score']}")
    print(f"  趋势方向: {info['trend_direction']}")

    # 执行多次检查
    print("\n🔄 执行健康检查...")
    results = []
    for i in range(15):
        status = checker.check_service_health("test_service")
        info = checker.get_service_adaptive_info("test_service")

        results.append({
            'check': i + 1,
            'status': status.value,
            'interval': info['current_interval'],
            'stability': info['stability_score'],
            'consecutive_failures': info['consecutive_failures']
        })

        print(f"  检查 {i + 1}: 状态={status.value}, 间隔={info['current_interval']:.1f}s, "
              f"稳定性={info['stability_score']:.2f}")

        time.sleep(0.1)  # 短暂延迟

    # 统计结果
    successful_checks = sum(1 for r in results if r['status'] == 'healthy')
    avg_interval = sum(r['interval'] for r in results) / len(results)
    final_stability = results[-1]['stability']

    print("\n📈 测试统计:")
    print(f"  总检查次数: {len(results)}")
    print(f"  成功检查数: {successful_checks}")
    print(".1f")
    print(".2f")
    print(".1f")
    # 获取自适应统计
    adaptive_stats = checker.get_adaptive_stats()
    print("\n🔧 自适应统计:")
    print(f"  总检查数: {adaptive_stats['total_checks']}")
    print(f"  自适应调整次数: {adaptive_stats['adaptive_adjustments']}")
    print(".1f")
    print(f"  间隔变化记录数: {len(adaptive_stats['interval_changes'])}")

    return {
        'total_checks': len(results),
        'successful_checks': successful_checks,
        'success_rate': successful_checks / len(results) * 100,
        'average_interval': avg_interval,
        'final_stability': final_stability,
        'adaptive_stats': adaptive_stats
    }


if __name__ == "__main__":
    print("自适应健康检查器测试...")

    # 运行测试
    test_results = test_adaptive_health_checker()

    print("\n✅ 自适应健康检查器测试完成")
    print(".1f")
    print(".1f")
    print(".2f")
    # 保存测试结果
    test_output = {
        'adaptive_health_check_test': test_results,
        'timestamp': datetime.now().isoformat(),
        'test_description': '自适应健康检查器功能测试'
    }

    with open('adaptive_health_check_test_results.json', 'w', encoding='utf - 8') as f:
        json.dump(test_output, f, ensure_ascii=False, indent=2, default=str)

    print("📁 测试结果已保存: adaptive_health_check_test_results.json")
