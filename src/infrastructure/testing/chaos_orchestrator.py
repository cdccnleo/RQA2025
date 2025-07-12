#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 混沌实验编排器
根据配置文件执行混沌实验
"""

import yaml
import logging
from datetime import datetime
from typing import List, Dict, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from .chaos_engine import ChaosEngine, ChaosError

logger = logging.getLogger(__name__)

class ChaosOrchestrator:
    """混沌实验编排器"""

    def __init__(self, config_path: str = "config/chaos_experiments.yaml"):
        """
        初始化编排器
        :param config_path: 配置文件路径
        """
        self.engine = ChaosEngine()
        self.scheduler = BackgroundScheduler()
        self.experiments = self._load_config(config_path)

    def _load_config(self, path: str) -> List[Dict]:
        """加载混沌实验配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('experiments', [])
        except Exception as e:
            logger.error(f"加载混沌配置失败: {str(e)}")
            return []

    def run_experiment_by_name(self, name: str) -> Optional[Dict]:
        """
        按名称执行混沌实验
        :param name: 实验名称
        :return: 执行结果报告
        """
        experiment = next(
            (exp for exp in self.experiments if exp['name'] == name and exp['enabled']),
            None
        )

        if not experiment:
            logger.warning(f"未找到可用的混沌实验: {name}")
            return None

        logger.info(f"🚀 开始执行混沌实验: {experiment['name']}")

        try:
            # 执行不同类型的实验
            if experiment['type'] == "network_partition":
                report = self.engine.simulate_network_partition(
                    duration=experiment['duration'],
                    target_services=experiment.get('target_services')
                )
            elif experiment['type'] == "fpga_failure":
                report = self.engine.simulate_fpga_failure(
                    duration=experiment['duration'],
                    failure_mode=experiment.get('mode', 'complete')
                )
            else:
                raise ChaosError(f"未知的实验类型: {experiment['type']}")

            # 记录实验结果
            result = {
                'experiment': experiment['name'],
                'timestamp': datetime.now().isoformat(),
                'duration': report.recovery_time,
                'success': report.is_success,
                'affected_components': report.affected_components
            }

            logger.info(f"✅ 混沌实验完成: {experiment['name']}")
            return result

        except ChaosError as e:
            logger.error(f"混沌实验执行失败: {experiment['name']}, 错误: {str(e)}")
            return {
                'experiment': experiment['name'],
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }

    def schedule_experiments(self) -> None:
        """调度所有启用的混沌实验"""
        for exp in self.experiments:
            if exp['enabled'] and 'schedule' in exp:
                self.scheduler.add_job(
                    self.run_experiment_by_name,
                    'cron',
                    args=[exp['name']],
                    **self._parse_schedule(exp['schedule'])
                )
                logger.info(f"已调度混沌实验: {exp['name']} ({exp['schedule']})")

        self.scheduler.start()

    def _parse_schedule(self, schedule_str: str) -> Dict:
        """解析cron格式的调度配置"""
        parts = schedule_str.split()
        if len(parts) != 5:
            return {'trigger': 'date'}  # 默认立即执行

        return {
            'minute': parts[0],
            'hour': parts[1],
            'day': parts[2],
            'month': parts[3],
            'day_of_week': parts[4]
        }

    def list_available_experiments(self) -> List[Dict]:
        """获取可用的混沌实验列表"""
        return [
            {
                'name': exp['name'],
                'type': exp['type'],
                'description': exp['description'],
                'enabled': exp['enabled']
            }
            for exp in self.experiments
        ]


if __name__ == '__main__':
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    orchestrator = ChaosOrchestrator()
    print("可用混沌实验:", orchestrator.list_available_experiments())

    # 执行指定实验
    result = orchestrator.run_experiment_by_name("网络分区测试")
    print("实验结果:", result)

    # 启动调度器(后台运行)
    orchestrator.schedule_experiments()
    input("按Enter键退出...\n")
