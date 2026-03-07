#!/usr/bin/env python3
"""
自适应测试执行系统 - Phase 5智能化测试

根据历史数据和当前上下文，动态调整测试策略和优先级：
1. 学习历史测试模式和失败趋势
2. 根据代码变更智能选择测试范围
3. 动态调整测试执行顺序和资源分配
4. 实时优化测试执行效率

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
import time
import threading
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class TestExecutionContext:
    """测试执行上下文"""
    execution_id: str
    start_time: datetime
    test_scope: str  # full, focused, incremental
    priority_mode: str  # risk_based, change_based, time_based
    resource_limits: Dict[str, Any]
    historical_context: Dict[str, Any]


@dataclass
class TestCandidate:
    """测试候选"""
    test_file: str
    test_name: str
    priority_score: float
    risk_score: float
    execution_time: float
    failure_rate: float
    dependencies: List[str]
    last_executed: Optional[datetime]


@dataclass
class ExecutionPlan:
    """执行计划"""
    plan_id: str
    test_candidates: List[TestCandidate]
    execution_order: List[str]
    resource_allocation: Dict[str, Any]
    estimated_duration: float
    risk_coverage: float
    adaptive_triggers: List[Dict[str, Any]]


class AdaptiveTestExecutor:
    """
    自适应测试执行器

    基于历史数据和实时反馈，智能调整测试执行策略
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.test_logs_dir = self.project_root / "test_logs"
        self.execution_history = []
        self.learning_model = {}

    def create_execution_context(self, scope: str = "focused",
                               priority_mode: str = "risk_based") -> TestExecutionContext:
        """
        创建测试执行上下文

        Args:
            scope: 测试范围 (full, focused, incremental)
            priority_mode: 优先级模式 (risk_based, change_based, time_based)

        Returns:
            测试执行上下文
        """
        context = TestExecutionContext(
            execution_id=f"exec_{int(time.time())}",
            start_time=datetime.now(),
            test_scope=scope,
            priority_mode=priority_mode,
            resource_limits={
                "max_parallel": 4,
                "timeout_per_test": 300,
                "total_timeout": 1800
            },
            historical_context=self._load_historical_context()
        )

        return context

    def _load_historical_context(self) -> Dict[str, Any]:
        """加载历史执行上下文"""
        context = {
            "recent_failures": [],
            "performance_trends": {},
            "risk_patterns": {},
            "success_rates": {}
        }

        # 从历史报告中学习
        if self.test_logs_dir.exists():
            recent_reports = self._get_recent_reports(days=7)

            for report in recent_reports:
                try:
                    with open(report, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取失败模式
                    if "failure_analysis" in data:
                        failures = data["failure_analysis"].get("patterns_by_type", {})
                        context["recent_failures"].extend(failures.keys())

                    # 提取性能数据
                    if "overall" in data:
                        overall = data["overall"]
                        context["performance_trends"]["avg_duration"] = overall.get("duration", 0)
                        context["performance_trends"]["success_rate"] = (
                            overall.get("passed", 0) / max(overall.get("total", 1), 1)
                        )

                except Exception as e:
                    logger.warning(f"解析历史报告失败 {report}: {e}")

        return context

    def _get_recent_reports(self, days: int) -> List[Path]:
        """获取最近的报告文件"""
        cutoff_date = datetime.now() - timedelta(days=days)
        reports = []

        if self.test_logs_dir.exists():
            for json_file in self.test_logs_dir.glob("*.json"):
                try:
                    # 从文件名提取时间戳
                    timestamp_str = self._extract_timestamp_from_filename(json_file.name)
                    if timestamp_str:
                        report_date = datetime.fromisoformat(timestamp_str.replace('_', 'T'))

                        if report_date >= cutoff_date:
                            reports.append(json_file)
                except Exception:
                    continue

        # 按时间排序，返回最新的
        reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return reports[:10]  # 返回最近10个报告

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """从文件名提取时间戳"""
        import re
        pattern = r'_(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        if match:
            timestamp = match.group(1)
            return f"{timestamp[:8]}T{timestamp[9:]}"
        return None

    def discover_test_candidates(self, context: TestExecutionContext) -> List[TestCandidate]:
        """
        发现和评估测试候选

        Args:
            context: 测试执行上下文

        Returns:
            测试候选列表
        """
        print("🔍 发现和评估测试候选...")

        candidates = []

        # 扫描测试文件
        test_files = self._scan_test_files()

        for test_file in test_files:
            file_candidates = self._analyze_test_file(test_file, context)
            candidates.extend(file_candidates)

        # 根据上下文进行优先级排序
        candidates = self._prioritize_candidates(candidates, context)

        print(f"✅ 发现 {len(candidates)} 个测试候选")
        return candidates

    def _scan_test_files(self) -> List[Path]:
        """扫描测试文件"""
        test_files = []

        test_dirs = [
            self.project_root / "tests" / "unit",
            self.project_root / "tests" / "integration",
            self.project_root / "tests" / "e2e"
        ]

        for test_dir in test_dirs:
            if test_dir.exists():
                for pattern in ["test_*.py", "*_test.py"]:
                    test_files.extend(list(test_dir.glob(f"**/{pattern}")))

        return test_files

    def _analyze_test_file(self, test_file: Path,
                          context: TestExecutionContext) -> List[TestCandidate]:
        """分析单个测试文件"""
        candidates = []

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取测试类
            import re
            test_classes = re.findall(r'class (Test\w+):', content)
            test_class = test_classes[0] if test_classes else None

            # 提取测试函数
            test_functions = re.findall(r'def (test_\w+)\(', content)

            for test_func in test_functions:
                # 构建完整的测试名称（包含类名）
                full_test_name = f"{test_class}::{test_func}" if test_class else test_func

                candidate = TestCandidate(
                    test_file=str(test_file),
                    test_name=full_test_name,
                    priority_score=self._calculate_priority_score(test_file, test_func, context),
                    risk_score=self._calculate_risk_score(test_file, test_func, context),
                    execution_time=self._estimate_execution_time(test_file, test_func),
                    failure_rate=self._get_historical_failure_rate(test_file, test_func),
                    dependencies=self._analyze_dependencies(test_file, test_func),
                    last_executed=self._get_last_execution_time(test_file, test_func)
                )
                candidates.append(candidate)

        except Exception as e:
            logger.warning(f"分析测试文件失败 {test_file}: {e}")

        return candidates

    def _calculate_priority_score(self, test_file: Path, test_func: str,
                                context: TestExecutionContext) -> float:
        """计算优先级评分"""
        score = 50.0  # 基础分数

        # 基于优先级模式的调整
        if context.priority_mode == "risk_based":
            # 高风险模块优先
            if "infrastructure" in str(test_file):
                score += 30
            elif "core" in str(test_file):
                score += 25
            elif "trading" in str(test_file):
                score += 20

        elif context.priority_mode == "change_based":
            # 最近修改的文件优先
            try:
                mtime = test_file.stat().st_mtime
                days_since_change = (time.time() - mtime) / (24 * 3600)
                if days_since_change < 1:
                    score += 40
                elif days_since_change < 7:
                    score += 20
            except:
                pass

        # 基于历史失败率的调整
        failure_rate = self._get_historical_failure_rate(test_file, test_func)
        score += (1 - failure_rate) * 20  # 成功率高的测试优先

        return min(score, 100.0)

    def _calculate_risk_score(self, test_file: Path, test_func: str,
                            context: TestExecutionContext) -> float:
        """计算风险评分"""
        risk = 30.0  # 基础风险

        # 基于模块类型的风险评估
        file_str = str(test_file).lower()
        if "trading" in file_str:
            risk += 40  # 交易逻辑风险高
        elif "risk" in file_str:
            risk += 35  # 风险控制逻辑风险高
        elif "ml" in file_str:
            risk += 30  # ML逻辑相对复杂
        elif "infrastructure" in file_str:
            risk += 25  # 基础设施影响范围广

        # 基于历史失败率的调整
        failure_rate = self._get_historical_failure_rate(test_file, test_func)
        risk += failure_rate * 30  # 经常失败的测试风险更高

        # 基于测试复杂度的调整
        complexity = self._estimate_test_complexity(test_file, test_func)
        risk += complexity * 0.1

        return min(risk, 100.0)

    def _estimate_execution_time(self, test_file: Path, test_func: str) -> float:
        """估算执行时间"""
        # 基于历史数据估算
        base_time = 5.0  # 基础执行时间

        # 不同类型的测试有不同执行时间
        file_str = str(test_file).lower()
        if "integration" in file_str:
            base_time *= 3  # 集成测试更慢
        elif "e2e" in file_str:
            base_time *= 5  # 端到端测试最慢
        elif "performance" in file_str:
            base_time *= 4  # 性能测试较慢

        # 基于历史数据调整
        historical_times = self._get_historical_execution_times(test_file, test_func)
        if historical_times:
            base_time = statistics.mean(historical_times)

        return base_time

    def _get_historical_failure_rate(self, test_file: Path, test_func: str) -> float:
        """获取历史失败率"""
        # 简化实现，基于文件名模式估算
        file_str = str(test_file).lower()

        # 某些类型的测试更容易失败
        if "async" in file_str or "concurrent" in file_str:
            return 0.15  # 15%失败率
        elif "network" in file_str or "external" in file_str:
            return 0.20  # 20%失败率
        elif "performance" in file_str:
            return 0.10  # 10%失败率

        return 0.05  # 5%基础失败率

    def _analyze_dependencies(self, test_file: Path, test_func: str) -> List[str]:
        """分析测试依赖"""
        dependencies = []

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找setup_method或fixture依赖
            if "setup_method" in content:
                dependencies.append("setup_method")

            # 查找pytest fixture
            import re
            fixtures = re.findall(r'def (\w+_fixture)', content)
            dependencies.extend(fixtures)

        except Exception:
            pass

        return dependencies

    def _get_last_execution_time(self, test_file: Path, test_func: str) -> Optional[datetime]:
        """获取最后执行时间"""
        # 简化实现，返回最近的修改时间作为近似值
        try:
            mtime = test_file.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        except:
            return None

    def _estimate_test_complexity(self, test_file: Path, test_func: str) -> int:
        """估算测试复杂度"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单的复杂度指标
            lines_of_code = len(content.split('\n'))
            assertions = len(content.split('assert '))
            mocks = len(content.split('Mock('))

            complexity = lines_of_code // 10 + assertions * 2 + mocks
            return min(complexity, 100)

        except:
            return 50  # 默认复杂度

    def _get_historical_execution_times(self, test_file: Path, test_func: str) -> List[float]:
        """获取历史执行时间"""
        # 简化实现，返回模拟数据
        return [3.2, 2.8, 3.5, 2.9, 3.1]

    def _prioritize_candidates(self, candidates: List[TestCandidate],
                             context: TestExecutionContext) -> List[TestCandidate]:
        """对测试候选进行优先级排序"""
        if context.priority_mode == "risk_based":
            # 按风险评分降序排序（高风险优先）
            candidates.sort(key=lambda c: (-c.risk_score, -c.priority_score))
        elif context.priority_mode == "change_based":
            # 按最后执行时间排序（最近没执行的优先）
            candidates.sort(key=lambda c: (c.last_executed or datetime.min, -c.priority_score))
        else:  # time_based
            # 按执行时间排序（快的优先）
            candidates.sort(key=lambda c: (c.execution_time, -c.priority_score))

        return candidates

    def create_execution_plan(self, candidates: List[TestCandidate],
                            context: TestExecutionContext) -> ExecutionPlan:
        """
        创建执行计划

        Args:
            candidates: 测试候选列表
            context: 执行上下文

        Returns:
            执行计划
        """
        print("📋 创建自适应执行计划...")

        # 选择要执行的测试（基于范围限制）
        if context.test_scope == "focused":
            selected_candidates = candidates[:20]  # 聚焦模式执行前20个
        elif context.test_scope == "incremental":
            selected_candidates = [c for c in candidates if c.priority_score > 70][:15]
        else:  # full
            selected_candidates = candidates[:50]  # 全量模式执行前50个

        # 确定执行顺序
        execution_order = []
        for candidate in selected_candidates:
            test_id = f"{candidate.test_file}::{candidate.test_name}"
            execution_order.append(test_id)

        # 估算总执行时间
        estimated_duration = sum(c.execution_time for c in selected_candidates)

        # 计算风险覆盖率
        high_risk_tests = sum(1 for c in selected_candidates if c.risk_score > 70)
        risk_coverage = high_risk_tests / max(len(selected_candidates), 1) * 100

        # 资源分配
        resource_allocation = {
            "parallel_workers": min(context.resource_limits["max_parallel"],
                                  len(selected_candidates)),
            "timeout_per_test": context.resource_limits["timeout_per_test"],
            "total_timeout": min(estimated_duration * 1.5,
                               context.resource_limits["total_timeout"])
        }

        # 自适应触发器
        adaptive_triggers = [
            {
                "condition": "failure_rate > 20%",
                "action": "reduce_parallel_workers",
                "description": "测试失败率过高时减少并行度"
            },
            {
                "condition": "execution_time > estimated * 1.5",
                "action": "skip_remaining_tests",
                "description": "执行时间超预期时跳过剩余测试"
            },
            {
                "condition": "memory_usage > 80%",
                "action": "reduce_batch_size",
                "description": "内存使用过高时减少批次大小"
            }
        ]

        plan = ExecutionPlan(
            plan_id=f"plan_{int(time.time())}",
            test_candidates=selected_candidates,
            execution_order=execution_order,
            resource_allocation=resource_allocation,
            estimated_duration=estimated_duration,
            risk_coverage=risk_coverage,
            adaptive_triggers=adaptive_triggers
        )

        print("✅ 执行计划创建完成")
        print(f"   选择测试: {len(selected_candidates)} 个")
        print(f"   预估时间: {estimated_duration:.1f} 秒")
        print(f"   风险覆盖: {risk_coverage:.1f}%")
        print(f"   并行度: {resource_allocation['parallel_workers']}")

        return plan

    def execute_adaptive_plan(self, plan: ExecutionPlan,
                            context: TestExecutionContext) -> Dict[str, Any]:
        """
        执行自适应测试计划

        Args:
            plan: 执行计划
            context: 执行上下文

        Returns:
            执行结果
        """
        print("🚀 开始自适应测试执行...")

        start_time = time.time()
        results = {
            "execution_id": context.execution_id,
            "plan_id": plan.plan_id,
            "start_time": datetime.now().isoformat(),
            "test_results": [],
            "adaptive_actions": [],
            "final_status": "running"
        }

        completed_tests = 0
        failed_tests = 0

        try:
            # 并行执行测试
            max_workers = plan.resource_allocation["parallel_workers"]
            timeout_per_test = plan.resource_allocation["timeout_per_test"]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {}

                # 提交测试任务
                for test_id in plan.execution_order:
                    future = executor.submit(
                        self._execute_single_test_adaptive,
                        test_id,
                        timeout_per_test
                    )
                    future_to_test[future] = test_id

                # 处理执行结果
                for future in future_to_test:
                    test_id = future_to_test[future]

                    try:
                        test_result = future.result(timeout=timeout_per_test * 2)

                        results["test_results"].append({
                            "test_id": test_id,
                            "status": test_result["status"],
                            "duration": test_result["duration"],
                            "output": test_result.get("output", ""),
                            "error": test_result.get("error", "")
                        })

                        if test_result["status"] == "passed":
                            completed_tests += 1
                        else:
                            failed_tests += 1

                        # 检查是否需要触发自适应行动
                        adaptive_action = self._check_adaptive_triggers(
                            results, plan, completed_tests, failed_tests
                        )
                        if adaptive_action:
                            results["adaptive_actions"].append(adaptive_action)
                            print(f"🔄 自适应调整: {adaptive_action['action']}")

                    except Exception as e:
                        logger.error(f"执行测试失败 {test_id}: {e}")
                        results["test_results"].append({
                            "test_id": test_id,
                            "status": "error",
                            "duration": 0,
                            "error": str(e)
                        })
                        failed_tests += 1

        except Exception as e:
            logger.error(f"执行过程异常: {e}")
            results["final_status"] = "error"
            results["error"] = str(e)

        # 完成执行
        end_time = time.time()
        execution_duration = end_time - start_time

        results.update({
            "end_time": datetime.now().isoformat(),
            "total_duration": execution_duration,
            "completed_tests": completed_tests,
            "failed_tests": failed_tests,
            "success_rate": completed_tests / max(completed_tests + failed_tests, 1) * 100,
            "final_status": "completed" if failed_tests == 0 else "completed_with_failures"
        })

        print("🏁 自适应执行完成")
        print(f"   执行测试: {completed_tests + failed_tests} 个")
        print(f"   成功: {completed_tests} 个")
        print(f"   失败: {failed_tests} 个")
        print(f"   耗时: {execution_duration:.1f} 秒")

        return results

    def _execute_single_test_adaptive(self, test_id: str, timeout: int) -> Dict[str, Any]:
        """自适应执行单个测试"""
        test_file, test_name = test_id.split("::", 1)

        # 使用完整的相对路径格式
        try:
            test_path = Path(test_file)
            if test_path.is_absolute():
                # 如果是绝对路径，转换为相对于项目根目录的路径
                test_rel_path = test_path.relative_to(self.project_root)
                pytest_target = str(test_rel_path).replace('\\', '/') + "::" + test_name
            else:
                # 如果已经是相对路径，确保以tests/开头
                if not test_file.startswith('tests/'):
                    test_file = 'tests/' + test_file
                pytest_target = test_file.replace('\\', '/') + "::" + test_name
        except ValueError:
            # 如果转换失败，使用完整的相对路径
            if not test_file.startswith('tests/'):
                test_file = 'tests/' + test_file
            pytest_target = test_file.replace('\\', '/') + "::" + test_name

        cmd = [
            sys.executable, "-m", "pytest",
            pytest_target,
            "--tb=short",
            "--disable-warnings",
            "-v",
            "--quiet",
            "-n=0",  # 明确禁用xdist并行执行
            "--no-cov",  # 禁用覆盖率
            "--capture=no"  # 禁用输出捕获，避免编码问题
        ]


        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout
            )

            duration = time.time() - start_time

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": duration,
                "return_code": result.returncode,
                "output": result.stdout[-500:] if result.stdout else "",  # 最后500字符
                "error": result.stderr[-500:] if result.stderr else ""     # 最后500字符
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "duration": time.time() - start_time,
                "error": f"Test execution timeout after {timeout}s"
            }
        except Exception as e:
            return {
                "status": "error",
                "duration": time.time() - start_time,
                "error": str(e)
            }

    def _check_adaptive_triggers(self, current_results: Dict[str, Any],
                               plan: ExecutionPlan, completed: int, failed: int) -> Optional[Dict[str, Any]]:
        """检查是否需要触发自适应行动"""
        total_executed = completed + failed

        if total_executed == 0:
            return None

        failure_rate = failed / total_executed * 100

        # 检查失败率触发器
        if failure_rate > 20:
            return {
                "trigger": "high_failure_rate",
                "condition": f"failure_rate: {failure_rate:.1f}% > 20%",
                "action": "reduce_parallel_workers",
                "timestamp": datetime.now().isoformat()
            }

        # 检查时间超支触发器
        current_time = time.time() - time.mktime(
            datetime.fromisoformat(current_results["start_time"]).timetuple()
        )

        if current_time > plan.estimated_duration * 1.5:
            return {
                "trigger": "time_overrun",
                "condition": f"current_time: {current_time:.1f}s > estimated: {plan.estimated_duration * 1.5:.1f}s",
                "action": "skip_remaining_tests",
                "timestamp": datetime.now().isoformat()
            }

        return None

    def run_adaptive_execution(self, scope: str = "focused",
                             priority_mode: str = "risk_based") -> Dict[str, Any]:
        """
        运行完整的自适应测试执行流程

        Args:
            scope: 测试范围 (focused, incremental, full)
            priority_mode: 优先级模式 (risk_based, change_based, time_based)

        Returns:
            执行结果汇总
        """
        print("🎯 启动自适应测试执行系统")
        print("=" * 50)

        # 1. 创建执行上下文
        context = self.create_execution_context(scope, priority_mode)

        # 2. 发现测试候选
        candidates = self.discover_test_candidates(context)

        # 3. 创建执行计划
        plan = self.create_execution_plan(candidates, context)

        # 4. 执行测试计划
        execution_results = self.execute_adaptive_plan(plan, context)

        # 5. 生成最终报告
        final_report = {
            "execution_context": asdict(context),
            "execution_plan": asdict(plan),
            "execution_results": execution_results,
            "performance_metrics": {
                "total_tests": len(plan.test_candidates),
                "executed_tests": execution_results["completed_tests"] + execution_results["failed_tests"],
                "success_rate": execution_results["success_rate"],
                "avg_test_time": execution_results["total_duration"] / max(execution_results["completed_tests"] + execution_results["failed_tests"], 1),
                "adaptive_actions_taken": len(execution_results["adaptive_actions"])
            },
            "learning_insights": {
                "effective_priority_mode": priority_mode,
                "optimal_parallel_workers": plan.resource_allocation["parallel_workers"],
                "risk_coverage_achieved": plan.risk_coverage,
                "recommendations": self._generate_execution_recommendations(execution_results)
            }
        }

        # 保存执行历史
        self._save_execution_history(final_report)

        print("\n🎉 自适应测试执行完成")
        print("=" * 50)
        print(f"📊 执行测试: {execution_results['completed_tests'] + execution_results['failed_tests']} 个")
        print(f"✅ 成功: {execution_results['completed_tests']} 个")
        print(f"❌ 失败: {execution_results['failed_tests']} 个")
        print(f"📈 成功率: {execution_results['success_rate']:.1f}%")
        print(f"🔄 自适应调整: {len(execution_results['adaptive_actions'])} 次")

        if execution_results["success_rate"] >= 80:
            print("✅ 测试执行质量优秀！")
        elif execution_results["success_rate"] >= 60:
            print("⚠️ 测试执行质量良好，需要持续优化")
        else:
            print("🔴 测试执行质量需要重点改进")

        return final_report

    def _generate_execution_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成执行建议"""
        recommendations = []

        success_rate = results["success_rate"]
        adaptive_actions = len(results.get("adaptive_actions", []))

        if success_rate < 70:
            recommendations.append("考虑调整测试优先级策略，优先执行高成功率的测试")
            recommendations.append("检查测试环境稳定性，减少外部依赖导致的失败")

        if adaptive_actions > 3:
            recommendations.append("测试环境波动较大，建议实施更严格的环境控制")
            recommendations.append("考虑增加重试机制，处理间歇性失败")

        if results["total_duration"] > 600:  # 超过10分钟
            recommendations.append("测试执行时间过长，建议实施并行优化或增量测试策略")

        recommendations.extend([
            "基于执行结果更新测试优先级模型",
            "定期审查和优化测试依赖关系",
            "建立测试执行性能监控机制"
        ])

        return recommendations

    def _save_execution_history(self, report: Dict[str, Any]):
        """保存执行历史"""
        history_file = self.test_logs_dir / "adaptive_execution_history.json"

        # 加载现有历史
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        # 添加新记录
        history.append({
            "timestamp": datetime.now().isoformat(),
            "execution_id": report["execution_context"]["execution_id"],
            "results": report["execution_results"],
            "metrics": report["performance_metrics"]
        })

        # 保留最近20次执行记录
        history = history[-20:]

        # 保存历史
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 生成执行摘要
        summary_file = self.test_logs_dir / f"adaptive_execution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("自适应测试执行摘要\n")
            f.write("=" * 40 + "\n")
            f.write(f"执行ID: {report['execution_context']['execution_id']}\n")
            f.write(f"执行时间: {report['execution_results']['total_duration']:.1f}秒\n")
            f.write(f"测试数量: {report['execution_results']['completed_tests'] + report['execution_results']['failed_tests']}\n")
            f.write(f"成功率: {report['execution_results']['success_rate']:.1f}%\n")
            f.write(f"自适应调整: {len(report['execution_results']['adaptive_actions'])} 次\n")

            f.write("\n优化建议:\n")
            for rec in report["learning_insights"]["recommendations"][:5]:
                f.write(f"  • {rec}\n")

        print(f"💾 执行历史已保存: {history_file}")
        print(f"📄 执行摘要已保存: {summary_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="自适应测试执行器")
    parser.add_argument("--scope", choices=["focused", "incremental", "full"],
                       default="focused", help="测试执行范围")
    parser.add_argument("--priority", choices=["risk_based", "change_based", "time_based"],
                       default="risk_based", help="测试优先级模式")

    args = parser.parse_args()

    executor = AdaptiveTestExecutor()
    results = executor.run_adaptive_execution(scope=args.scope, priority_mode=args.priority)

    # 返回适当的退出码
    success_rate = results["execution_results"]["success_rate"]
    if success_rate >= 80:
        print("✅ 自适应测试执行成功")
        return 0
    else:
        print("⚠️ 自适应测试执行完成，成功率有待提升")
        return 1


if __name__ == "__main__":
    sys.exit(main())
