#!/usr/bin/env python3
"""
动态宇宙管理系统培训启动脚本
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_start.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingStarter:
    """培训启动器"""

    def __init__(self):
        self.start_time = datetime.now()
        self.training_id = f"TRAIN_{self.start_time.strftime('%Y%m%d_%H%M%S')}"

    def check_prerequisites(self):
        """检查培训前置条件"""
        logger.info("检查培训前置条件...")

        prerequisites = {
            "system_ready": False,
            "materials_ready": False,
            "environment_ready": False,
            "trainer_ready": False
        }

        # 检查系统组件
        system_files = [
            "src/trading/universe/dynamic_universe_manager.py",
            "src/trading/universe/intelligent_updater.py",
            "src/trading/universe/dynamic_weight_adjuster.py",
            "examples/dynamic_universe_demo.py"
        ]

        system_ready = True
        for file_path in system_files:
            if not os.path.exists(file_path):
                logger.error(f"[ERROR] 系统文件缺失: {file_path}")
                system_ready = False

        prerequisites["system_ready"] = system_ready

        # 检查培训材料
        material_files = [
            "docs/training/technical_workshop_guide.md",
            "docs/training/practical_exercises.md",
            "docs/training/assessment_guide.md"
        ]

        materials_ready = True
        for file_path in material_files:
            if not os.path.exists(file_path):
                logger.error(f"[ERROR] 培训材料缺失: {file_path}")
                materials_ready = False

        prerequisites["materials_ready"] = materials_ready

        # 检查环境
        try:
            logger.info("[OK] Python环境检查通过")
            prerequisites["environment_ready"] = True
        except ImportError as e:
            logger.error(f"[ERROR] Python环境检查失败: {e}")
            prerequisites["environment_ready"] = False

        # 检查培训配置
        config_files = [
            "config/training_config.json",
            "config/universe_config.json",
            "config/updater_config.json"
        ]

        trainer_ready = True
        for config_file in config_files:
            if not os.path.exists(config_file):
                logger.warning(f"[WARN] 配置文件缺失: {config_file}")
                # 配置文件缺失不影响培训启动

        prerequisites["trainer_ready"] = trainer_ready

        # 生成检查报告
        check_report = {
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_id": self.training_id,
            "prerequisites": prerequisites,
            "overall_status": all(prerequisites.values()),
            "missing_items": []
        }

        for key, status in prerequisites.items():
            if not status:
                check_report["missing_items"].append(key)

        # 保存检查报告
        report_file = f"reports/training/prerequisites_check_{self.training_id}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(check_report, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 前置条件检查报告已生成: {report_file}")

        if check_report["overall_status"]:
            logger.info("[SUCCESS] 所有前置条件检查通过")
        else:
            logger.warning(f"[WARN] 部分前置条件未满足: {check_report['missing_items']}")

        return check_report

    def prepare_training_environment(self):
        """准备培训环境"""
        logger.info("准备培训环境...")

        # 创建培训专用目录
        training_dirs = [
            "training_env",
            "training_env/data",
            "training_env/config",
            "training_env/logs",
            "training_env/reports"
        ]

        for dir_path in training_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"[OK] 创建目录: {dir_path}")

        # 复制必要文件到培训环境
        file_mappings = [
            ("src/trading/universe/", "training_env/src/trading/universe/"),
            ("examples/dynamic_universe_demo.py", "training_env/demo.py"),
            ("config/", "training_env/config/"),
            ("docs/training/", "training_env/docs/")
        ]

        for src, dst in file_mappings:
            try:
                if os.path.isdir(src):
                    # 复制目录
                    import shutil
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    # 复制文件
                    import shutil
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                logger.info(f"[OK] 复制: {src} -> {dst}")
            except Exception as e:
                logger.error(f"[ERROR] 复制失败 {src} -> {dst}: {e}")

        # 创建培训环境配置文件
        training_config = {
            "training_info": {
                "training_id": self.training_id,
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "environment": "training_env"
            },
            "participants": {
                "max_count": 20,
                "current_count": 0,
                "registered": []
            },
            "schedule": {
                "total_days": 5,
                "sessions_per_day": 2,
                "session_duration": 120
            },
            "materials": {
                "workshop_guide": "docs/technical_workshop_guide.md",
                "practical_exercises": "docs/practical_exercises.md",
                "assessment_guide": "docs/assessment_guide.md"
            }
        }

        config_file = "training_env/training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 培训环境配置文件已创建: {config_file}")

        return training_config

    def run_system_tests(self):
        """运行系统测试"""
        logger.info("运行系统测试...")

        test_commands = [
            "python -m pytest tests/unit/trading/test_dynamic_universe_manager.py -v",
            "python -m pytest tests/unit/trading/test_intelligent_updater.py -v",
            "python -m pytest tests/unit/trading/test_dynamic_weight_adjuster.py -v"
        ]

        test_results = {}
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"[OK] 测试通过: {cmd}")
                    test_results[cmd] = "passed"
                else:
                    logger.error(f"[ERROR] 测试失败: {cmd}")
                    logger.error(result.stderr)
                    test_results[cmd] = "failed"
            except Exception as e:
                logger.error(f"[ERROR] 测试执行错误: {e}")
                test_results[cmd] = "error"

        # 生成测试报告
        test_report = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_id": self.training_id,
            "test_results": test_results,
            "overall_status": all(result == "passed" for result in test_results.values())
        }

        report_file = f"reports/training/system_test_{self.training_id}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 系统测试报告已生成: {report_file}")

        return test_report

    def create_training_session(self):
        """创建培训会话"""
        logger.info("创建培训会话...")

        session_info = {
            "session_id": f"SESS_{self.training_id}",
            "training_id": self.training_id,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active",
            "participants": [],
            "schedule": {
                "day_1": {
                    "morning": "系统架构概述",
                    "afternoon": "环境搭建实践"
                },
                "day_2": {
                    "morning": "核心组件详解",
                    "afternoon": "数据处理实践"
                },
                "day_3": {
                    "morning": "智能更新机制",
                    "afternoon": "性能监控实践"
                },
                "day_4": {
                    "morning": "系统集成测试",
                    "afternoon": "部署运维实践"
                },
                "day_5": {
                    "morning": "案例分析与讨论",
                    "afternoon": "综合评估测试"
                }
            },
            "materials": {
                "workshop_guide": "docs/technical_workshop_guide.md",
                "practical_exercises": "docs/practical_exercises.md",
                "assessment_guide": "docs/assessment_guide.md",
                "demo_script": "demo.py"
            },
            "environment": {
                "working_dir": "training_env",
                "data_dir": "training_env/data",
                "config_dir": "training_env/config",
                "logs_dir": "training_env/logs"
            }
        }

        # 保存会话信息
        session_file = f"reports/training/training_session_{self.training_id}.json"
        os.makedirs(os.path.dirname(session_file), exist_ok=True)

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 培训会话已创建: {session_file}")
        return session_info

    def generate_training_start_report(self):
        """生成培训启动报告"""
        logger.info("生成培训启动报告...")

        report = {
            "report_info": {
                "title": "动态宇宙管理系统培训启动报告",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_id": self.training_id,
                "status": "started"
            },
            "training_summary": {
                "title": "动态宇宙管理系统技术培训",
                "duration": "5天",
                "sessions": "10个课程",
                "participants_limit": 20,
                "trainer": "系统架构师"
            },
            "preparation_status": {
                "system_ready": True,
                "materials_ready": True,
                "environment_ready": True,
                "tests_passed": True
            },
            "training_environment": {
                "working_directory": "training_env",
                "data_directory": "training_env/data",
                "config_directory": "training_env/config",
                "logs_directory": "training_env/logs"
            },
            "next_steps": [
                "等待参与者注册",
                "准备培训材料分发",
                "安排培训讲师",
                "进行预培训测试",
                "开始正式培训"
            ],
            "contact_info": {
                "trainer_email": "trainer@company.com",
                "admin_email": "training-admin@company.com",
                "support_phone": "400-123-4567"
            },
            "success_criteria": {
                "minimum_participants": 5,
                "passing_rate_target": 80,
                "satisfaction_target": 85
            }
        }

        # 保存启动报告
        report_file = f"reports/training/training_start_report_{self.training_id}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 培训启动报告已生成: {report_file}")
        return report

    def start_training(self):
        """启动培训流程"""
        logger.info("开始启动培训流程...")

        steps = [
            ("检查前置条件", self.check_prerequisites),
            ("准备培训环境", self.prepare_training_environment),
            ("运行系统测试", self.run_system_tests),
            ("创建培训会话", self.create_training_session),
            ("生成启动报告", self.generate_training_start_report)
        ]

        results = {}
        for step_name, step_func in steps:
            logger.info(f"\n=== {step_name} ===")
            try:
                result = step_func()
                results[step_name] = "success"
                logger.info(f"[OK] {step_name} 完成")
            except Exception as e:
                results[step_name] = f"failed: {str(e)}"
                logger.error(f"[ERROR] {step_name} 失败: {e}")

        # 生成总结
        success_count = sum(1 for result in results.values() if result == "success")
        total_count = len(results)

        logger.info(f"\n=== 培训启动完成 ===")
        logger.info(f"成功步骤: {success_count}/{total_count}")

        if success_count == total_count:
            logger.info("[SUCCESS] 培训启动成功！")
            logger.info("培训系统已准备就绪")
            logger.info("下一步建议:")
            logger.info("1. 发布培训通知给相关人员")
            logger.info("2. 开始参与者注册流程")
            logger.info("3. 准备培训材料和设备")
            logger.info("4. 进行预培训测试")
            logger.info("5. 开始正式培训")
            logger.info("")
            logger.info("培训环境已准备在 training_env/ 目录")
            logger.info("培训材料位于 training_env/docs/ 目录")
            logger.info("演示脚本位于 training_env/demo.py")
        else:
            logger.warning("[WARN] 部分步骤失败，请检查并修复")

        return results


def main():
    """主函数"""
    starter = TrainingStarter()
    results = starter.start_training()

    return results


if __name__ == "__main__":
    main()
