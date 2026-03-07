"""
基础设施层投产监控设置脚本

用途：建立完善的监控机制，确保投产后系统稳定性
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import psutil
import time


class InfrastructureMonitor:
    """基础设施层监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.alerts = []
        
    def setup_monitoring(self):
        """设置监控"""
        self.logger.info("设置基础设施层监控...")
        
        # 1. 系统性能监控
        self._setup_performance_monitoring()
        
        # 2. 错误率监控
        self._setup_error_monitoring()
        
        # 3. API响应时间监控
        self._setup_api_monitoring()
        
        # 4. 缓存命中率监控
        self._setup_cache_monitoring()
        
        # 5. 日志监控
        self._setup_log_monitoring()
        
        self.logger.info("监控设置完成")
    
    def _setup_performance_monitoring(self):
        """设置性能监控"""
        self.logger.info("设置性能监控...")
        
        # CPU使用率监控
        self.metrics['cpu_threshold'] = 80  # 80%告警
        
        # 内存使用率监控
        self.metrics['memory_threshold'] = 85  # 85%告警
        
        # 磁盘使用率监控
        self.metrics['disk_threshold'] = 90  # 90%告警
    
    def _setup_error_monitoring(self):
        """设置错误监控"""
        self.logger.info("设置错误监控...")
        
        # 错误率阈值
        self.metrics['error_rate_threshold'] = 1.0  # 1%告警
        
        # 致命错误阈值
        self.metrics['critical_error_threshold'] = 0  # 任何致命错误都告警
    
    def _setup_api_monitoring(self):
        """设置API监控"""
        self.logger.info("设置API监控...")
        
        # API响应时间阈值
        self.metrics['api_response_threshold'] = 1000  # 1秒告警
        
        # API错误率阈值
        self.metrics['api_error_rate_threshold'] = 2.0  # 2%告警
    
    def _setup_cache_monitoring(self):
        """设置缓存监控"""
        self.logger.info("设置缓存监控...")
        
        # 缓存命中率阈值
        self.metrics['cache_hit_rate_threshold'] = 70  # 低于70%告警
    
    def _setup_log_monitoring(self):
        """设置日志监控"""
        self.logger.info("设置日志监控...")
        
        # 日志错误关键词
        self.metrics['log_error_keywords'] = [
            'ERROR',
            'CRITICAL',
            'FATAL',
            'Exception',
            'Failed'
        ]
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'metrics': {},
            'alerts': []
        }
        
        # 检查CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        health_status['metrics']['cpu_percent'] = cpu_percent
        if cpu_percent > self.metrics['cpu_threshold']:
            alert = {
                'type': 'CPU_HIGH',
                'severity': 'WARNING',
                'message': f'CPU使用率过高: {cpu_percent}%',
                'threshold': self.metrics['cpu_threshold']
            }
            health_status['alerts'].append(alert)
            health_status['overall_status'] = 'warning'
        
        # 检查内存
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        health_status['metrics']['memory_percent'] = memory_percent
        if memory_percent > self.metrics['memory_threshold']:
            alert = {
                'type': 'MEMORY_HIGH',
                'severity': 'WARNING',
                'message': f'内存使用率过高: {memory_percent}%',
                'threshold': self.metrics['memory_threshold']
            }
            health_status['alerts'].append(alert)
            health_status['overall_status'] = 'warning'
        
        # 检查磁盘
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        health_status['metrics']['disk_percent'] = disk_percent
        if disk_percent > self.metrics['disk_threshold']:
            alert = {
                'type': 'DISK_HIGH',
                'severity': 'CRITICAL',
                'message': f'磁盘使用率过高: {disk_percent}%',
                'threshold': self.metrics['disk_threshold']
            }
            health_status['alerts'].append(alert)
            health_status['overall_status'] = 'critical'
        
        return health_status
    
    def monitor_continuously(self, interval_seconds: int = 60):
        """持续监控"""
        self.logger.info(f"开始持续监控（每{interval_seconds}秒）...")
        
        try:
            while True:
                health = self.check_system_health()
                
                # 记录健康状态
                self.logger.info(f"健康检查: {health['overall_status']}")
                
                # 如果有告警，记录详情
                if health['alerts']:
                    for alert in health['alerts']:
                        self.logger.warning(f"告警: {alert['message']}")
                
                # 等待下一次检查
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("监控已停止")


class DeploymentMonitoringSetup:
    """投产监控设置"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = InfrastructureMonitor()
    
    def setup_pre_deployment(self):
        """投产前监控设置"""
        self.logger.info("执行投产前监控设置...")
        
        # 1. 建立基线指标
        baseline = self._establish_baseline()
        self.logger.info(f"基线指标: {baseline}")
        
        # 2. 设置监控
        self.monitor.setup_monitoring()
        
        # 3. 验证监控系统
        self._verify_monitoring_system()
        
        self.logger.info("投产前监控设置完成")
        
        return True
    
    def _establish_baseline(self) -> Dict[str, Any]:
        """建立基线指标"""
        self.logger.info("建立基线指标...")
        
        baseline = {
            'cpu_baseline': psutil.cpu_percent(interval=5),
            'memory_baseline': psutil.virtual_memory().percent,
            'disk_baseline': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
        
        return baseline
    
    def _verify_monitoring_system(self):
        """验证监控系统"""
        self.logger.info("验证监控系统...")
        
        # 执行一次健康检查
        health = self.monitor.check_system_health()
        
        if health['overall_status'] in ['healthy', 'warning']:
            self.logger.info("监控系统正常")
        else:
            self.logger.error(f"监控系统异常: {health['overall_status']}")
            raise Exception("监控系统验证失败")


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("基础设施层投产监控设置")
    logger.info("=" * 80)
    
    # 创建监控设置
    setup = DeploymentMonitoringSetup()
    
    # 执行投产前设置
    try:
        success = setup.setup_pre_deployment()
        if success:
            logger.info("✅ 监控设置成功！")
            
            # 显示当前健康状态
            health = setup.monitor.check_system_health()
            logger.info(f"当前系统健康状态: {health['overall_status']}")
            logger.info(f"CPU: {health['metrics'].get('cpu_percent', 'N/A')}%")
            logger.info(f"内存: {health['metrics'].get('memory_percent', 'N/A')}%")
            logger.info(f"磁盘: {health['metrics'].get('disk_percent', 'N/A')}%")
            
            if health['alerts']:
                logger.warning("⚠️ 存在告警:")
                for alert in health['alerts']:
                    logger.warning(f"  - {alert['message']}")
        else:
            logger.error("❌ 监控设置失败！")
    except Exception as e:
        logger.error(f"❌ 监控设置异常: {e}")
        raise


if __name__ == "__main__":
    main()

















