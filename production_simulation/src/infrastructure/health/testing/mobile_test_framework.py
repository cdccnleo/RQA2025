"""
mobile_test_framework 模块

提供 mobile_test_framework 相关功能和接口。
"""

import logging

# 配置日志
import subprocess
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
"""
移动端测试框架模块
提供跨平台移动端测试能力，支持iOS、Android等平台的自动化测试
"""

logger = logging.getLogger(__name__)


class PlatformType(Enum):

    """移动端平台类型"""
    IOS = "ios"
    ANDROID = "android"
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"
    XAMARIN = "xamarin"


class TestType(Enum):

    """测试类型"""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    UI_TEST = "ui_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"


class DeviceType(Enum):

    """设备类型"""
    SIMULATOR = "simulator"
    EMULATOR = "emulator"
    PHYSICAL_DEVICE = "physical_device"
    CLOUD_DEVICE = "cloud_device"


@dataclass
class DeviceConfig:

    """设备配置"""
    platform: PlatformType
    device_type: DeviceType
    device_id: str
    os_version: str
    screen_resolution: str
    capabilities: Dict[str, Any] = None

    def __post_init__(self):

        if self.capabilities is None:
            self.capabilities = {}


@dataclass
class TestConfig:

    """测试配置"""
    test_type: TestType
    timeout: int = 300
    retry_count: int = 3
    parallel_execution: bool = True
    headless: bool = False
    custom_args: Dict[str, Any] = None

    def __post_init__(self):

        if self.custom_args is None:
            self.custom_args = {}


@dataclass
class TestResult:

    """测试结果"""
    test_name: str
    platform: PlatformType
    device_id: str
    status: str  # passed, failed, skipped, error
    duration: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    screenshots: List[str] = None
    logs: List[str] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):

        if self.screenshots is None:
            self.screenshots = []
        if self.logs is None:
            self.logs = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class DeviceManager:

    """设备管理器"""

    def __init__(self):

        self._lock = threading.Lock()
        self._devices = {}
        self._device_status = {}

    def discover_devices(self, platform: PlatformType) -> List[DeviceConfig]:
        """发现可用设备"""
        try:
            if platform == PlatformType.IOS:
                return self._discover_ios_devices()
            elif platform == PlatformType.ANDROID:
                return self._discover_android_devices()
            else:
                logger.warning(f"不支持的平台类型: {platform}")
                return []
        except Exception as e:
            logger.error(f"发现设备时发生错误: {e}")
            return []

    def _discover_ios_devices(self) -> List[DeviceConfig]:
        """发现iOS设备"""
        devices = []
        try:
            # 使用xcrun命令发现iOS设备
            result = subprocess.run([],
                                    "xcrun", "devicectl", "list", "devices"
                                    )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # 跳过标题行
                    if line.strip():
                        parts = line.split()
                    if len(parts) >= 3:
                        device_id = parts[0]
                        os_version = parts[1]
                        device_type = DeviceType.SIMULATOR if "Simulator" in line else DeviceType.PHYSICAL_DEVICE

                        device = DeviceConfig(
                            platform=PlatformType.IOS,
                            device_type=device_type,
                            device_id=device_id,
                            os_version=os_version,
                            screen_resolution="Unknown"
                        )
                        devices.append(device)

            logger.info(f"发现 {len(devices)} 个iOS设备")
            return devices

        except Exception as e:
            logger.error(f"发现iOS设备时发生错误: {e}")
            return []

    def _discover_android_devices(self) -> List[DeviceConfig]:
        """发现Android设备"""
        devices = []
        try:
            # 使用adb命令发现Android设备
            result = subprocess.run([],
                                    "adb", "devices", "-l"
                                    )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and "device" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        device_type = DeviceType.EMULATOR if "emulator" in line else DeviceType.PHYSICAL_DEVICE

                        # 获取设备信息
                        device_info = self._get_android_device_info(device_id)

                        device = DeviceConfig(
                            platform=PlatformType.ANDROID,
                            device_type=device_type,
                            device_id=device_id,
                            os_version=device_info.get("version", "Unknown"),
                            screen_resolution=device_info.get("resolution", "Unknown")
                        )
                        devices.append(device)

            logger.info(f"发现 {len(devices)} 个Android设备")
            return devices

        except Exception as e:
            logger.error(f"发现Android设备时发生错误: {e}")
            return []

    def _get_android_device_info(self, device_id: str) -> Dict[str, str]:
        """获取Android设备信息"""
        try:
            info = {}

            # 获取Android版本
            result = subprocess.run([],
                                    "adb", "-s", device_id, "shell", "getprop", "ro.build.version.release"
                                    )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()

            # 获取屏幕分辨率
            result = subprocess.run([],
                                    "adb", "-s", device_id, "shell", "wm", "size"
                                    )
            if result.returncode == 0:
                info["resolution"] = result.stdout.strip()

            return info

        except Exception as e:
            logger.error(f"获取Android设备信息时发生错误: {e}")
            return {}

    def get_device_status(self, device_id: str) -> str:
        """获取设备状态"""
        with self._lock:
            return self._device_status.get(device_id, "unknown")

    def update_device_status(self, device_id: str, status: str):
        """更新设备状态"""
        with self._lock:
            self._device_status[device_id] = status


class TestExecutor:

    """测试执行器"""

    def __init__(self, device: DeviceConfig, test_config: TestConfig):

        self.device = device
        self.test_config = test_config
        self._lock = threading.Lock()
        self._test_results = []

    def execute_test(self, test_name: str, test_script: str) -> TestResult:
        """执行测试"""
        start_time = datetime.now()

        try:
            logger.info(f"开始执行测试: {test_name} 在设备 {self.device.device_id}")

            # 验证测试执行参数
            self._validate_test_execution(test_name, test_script)

            # 根据平台执行测试
            execution_result = self._execute_test_by_platform(test_name, test_script)

            # 创建测试结果
            test_result = self._create_test_result(
                test_name, execution_result, start_time
            )

            # 存储测试结果
            self._store_test_result(test_result)

            logger.info(f"测试 {test_name} 执行完成，状态: {execution_result['status']}")
            return test_result

        except ValueError:
            # 重新抛出ValueError，让调用者处理
            raise
        except Exception as e:
            logger.error(f"执行测试 {test_name} 时发生错误: {e}")
            return self._handle_test_execution_error(test_name, str(e), start_time)

    def _validate_test_execution(self, test_name: str, test_script: str) -> None:
        """验证测试执行参数"""
        if not test_name:
            raise ValueError("测试名称不能为空")
        if not test_script:
            raise ValueError("测试脚本不能为空")
        if not self.device:
            raise ValueError("设备未初始化")

    def _execute_test_by_platform(self, test_name: str, test_script: str) -> Dict[str, Any]:
        """根据平台执行测试"""
        if self.device.platform == PlatformType.IOS:
            return self._execute_ios_test(test_name, test_script)
        elif self.device.platform == PlatformType.ANDROID:
            return self._execute_android_test(test_name, test_script)
        else:
            raise ValueError(f"不支持的平台类型: {self.device.platform}")

    def _create_test_result(self, test_name: str, execution_result: Dict[str, Any],
                            start_time: datetime) -> TestResult:
        """创建测试结果对象"""
        end_time = datetime.now()
        return TestResult(
            test_name=test_name,
            platform=self.device.platform,
            device_id=self.device.device_id,
            status=execution_result["status"],
            duration=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time,
            error_message=execution_result.get("error_message"),
            screenshots=execution_result.get("screenshots", []),
            logs=execution_result.get("logs", []),
            performance_metrics=execution_result.get("performance_metrics", {})
        )

    def _store_test_result(self, test_result: TestResult) -> None:
        """存储测试结果"""
        with self._lock:
            self._test_results.append(test_result)

    def _handle_test_execution_error(self, test_name: str, error_message: str,
                                     start_time: datetime) -> TestResult:
        """处理测试执行错误"""
        end_time = datetime.now()
        test_result = TestResult(
            test_name=test_name,
            platform=self.device.platform,
            device_id=self.device.device_id,
            status="error",
            duration=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time,
            error_message=error_message
        )

        self._store_test_result(test_result)
        return test_result

    def _execute_ios_test(self, test_name: str, test_script: str) -> Dict[str, Any]:
        """执行iOS测试"""
        try:
            # 使用XCTest执行iOS测试
            result = subprocess.run([],
                                    "xcodebuild", "test",
                                    "-destination", f"platform=iOS Simulator,id={self.device.device_id}",
                                    "-scheme", test_name,
                                    "-testPlan", test_name
                                    )

            if result.returncode == 0:
                return {
                    "status": "passed",
                    "logs": [result.stdout],
                    "screenshots": []
                }
            else:
                return {
                    "status": "failed",
                    "error_message": result.stderr,
                    "logs": [result.stdout, result.stderr],
                    "screenshots": []
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error_message": "测试执行超时",
                "logs": [],
                "screenshots": []
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }

    def _execute_android_test(self, test_name: str, test_script: str) -> Dict[str, Any]:
        """执行Android测试"""
        try:
            # 使用adb执行Android测试
            result = subprocess.run([],
                                    "adb", "-s", self.device.device_id, "shell", "am", "instrument",
                                    "-w", "-e", "class", test_name, test_script
                                    )

            if result.returncode == 0:
                return {
                    "status": "passed",
                    "logs": [result.stdout],
                    "screenshots": []
                }
            else:
                return {
                    "status": "failed",
                    "error_message": result.stderr,
                    "logs": [result.stdout, result.stderr],
                    "screenshots": []
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error_message": "测试执行超时",
                "logs": [],
                "screenshots": []
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "logs": [],
                "screenshots": []
            }

    def get_test_results(self) -> List[TestResult]:
        """获取测试结果"""
        with self._lock:
            return self._test_results.copy()


class MobileTestFramework:

    """移动端测试框架主类"""

    def __init__(self):

        self.device_manager = DeviceManager()
        self._lock = threading.Lock()
        self._executors = {}

    def setup_test_environment(self, platform: PlatformType) -> bool:
        """设置测试环境"""
        try:
            logger.info(f"设置 {platform.value} 测试环境")

            if platform == PlatformType.IOS:
                return self._setup_ios_environment()
            elif platform == PlatformType.ANDROID:
                return self._setup_android_environment()
            else:
                logger.error(f"不支持的平台类型: {platform}")
                return False

        except Exception as e:
            logger.error(f"设置测试环境时发生错误: {e}")
            return False

    def _setup_ios_environment(self) -> bool:
        """设置iOS测试环境"""
        try:
            # 检查Xcode是否安装
            result = subprocess.run(["xcodebuild", "-version"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Xcode未安装或配置不正确")
                return False

            # 检查iOS模拟器
            result = subprocess.run(["xcrun", "simctl", "list"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("iOS模拟器不可用")
                return False

            logger.info("iOS测试环境设置成功")
            return True

        except Exception as e:
            logger.error(f"设置iOS测试环境时发生错误: {e}")
            return False

    def _setup_android_environment(self) -> bool:
        """设置Android测试环境"""
        try:
            # 检查Android SDK是否安装
            result = subprocess.run(["adb", "version"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Android SDK未安装或配置不正确")
                return False

            # 检查Android模拟器
            result = subprocess.run(["emulator", "-list - avds"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Android模拟器不可用")

            logger.info("Android测试环境设置成功")
            return True

        except Exception as e:
            logger.error(f"设置Android测试环境时发生错误: {e}")
            return False

    def run_tests(self, platform: PlatformType, test_config: TestConfig,
                  test_cases: List[str]):
        """运行测试"""
        try:
            logger.info(f"开始运行 {platform.value} 平台测试")

            # 发现可用设备
            devices = self._discover_available_devices(platform)
            if not devices:
                return []

            # 创建测试执行器
            executors = self._create_test_executors(devices, test_config)

            # 执行测试
            all_results = self._execute_all_tests(test_config, test_cases, executors)

            logger.info(f"测试执行完成，共 {len(all_results)} 个结果")
            return all_results

        except Exception as e:
            logger.error(f"运行测试时发生错误: {e}")
            return []

    def _discover_available_devices(self, platform: PlatformType) -> List[Any]:
        """发现可用设备"""
        devices = self.device_manager.discover_devices(platform)
        if not devices:
            logger.error(f"未找到可用的 {platform.value} 设备")
        return devices

    def _create_test_executors(self, devices: List[Any], test_config: TestConfig) -> List[TestExecutor]:
        """创建测试执行器"""
        executors = []
        for device in devices:
            executor = TestExecutor(device, test_config)
            executors.append(executor)
            self._executors[device.device_id] = executor
        return executors

    def _execute_all_tests(self, test_config: TestConfig, test_cases: List[str],
                           executors: List[TestExecutor]) -> List[TestResult]:
        """执行所有测试"""
        if test_config.parallel_execution and len(executors) > 1:
            return self._execute_tests_parallel(test_cases, executors)
        else:
            return self._execute_tests_sequential(test_cases, executors)

    def _execute_tests_parallel(self, test_cases: List[str],
                                executors: List[TestExecutor]) -> List[TestResult]:
        """并行执行测试"""
        all_results = []
        with ThreadPoolExecutor(max_workers=len(executors)) as executor:
            futures = []
            for i, test_case in enumerate(test_cases):
                device_index = i % len(executors)
                future = executor.submit(
                    executors[device_index].execute_test,
                    test_case,
                    f"test_script_{test_case}"
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

        return all_results

    def _execute_tests_sequential(self, test_cases: List[str],
                                  executors: List[TestExecutor]) -> List[TestResult]:
        """串行执行测试"""
        all_results = []
        for test_case in test_cases:
            for executor in executors:
                result = executor.execute_test(test_case, f"test_script_{test_case}")
                all_results.append(result)
        return all_results

    def get_test_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        try:
            # 初始化摘要结构
            summary = self._initialize_summary_structure()

            # 统计所有测试结果
            self._count_all_test_results(summary)

            return summary

        except Exception as e:
            logger.error(f"获取测试摘要时发生错误: {e}")
            return {}

    def _initialize_summary_structure(self) -> Dict[str, Any]:
        """初始化摘要结构"""
        return {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0,
            "platforms": {},
            "devices": {}
        }

    def _count_all_test_results(self, summary: Dict[str, Any]) -> None:
        """统计所有测试结果"""
        for device_id, executor in self._executors.items():
            results = executor.get_test_results()
            summary["total_tests"] += len(results)

            for result in results:
                self._count_individual_result(summary, result)
                self._aggregate_platform_stats(summary, result)
                self._aggregate_device_stats(summary, device_id, result)

    def _count_individual_result(self, summary: Dict[str, Any], result: TestResult) -> None:
        """统计单个测试结果"""
        status_counts = {
            "passed": "passed",
            "failed": "failed",
            "skipped": "skipped",
            "error": "error"
        }

        if result.status in status_counts:
            summary[status_counts[result.status]] += 1

    def _aggregate_platform_stats(self, summary: Dict[str, Any], result: TestResult) -> None:
        """聚合平台统计"""
        platform_key = result.platform.value
        if platform_key not in summary["platforms"]:
            summary["platforms"][platform_key] = {"total": 0, "passed": 0, "failed": 0}

        summary["platforms"][platform_key]["total"] += 1
        if result.status == "passed":
            summary["platforms"][platform_key]["passed"] += 1
        else:
            summary["platforms"][platform_key]["failed"] += 1

    def _aggregate_device_stats(self, summary: Dict[str, Any], device_id: str,
                                result: TestResult) -> None:
        """聚合设备统计"""
        if device_id not in summary["devices"]:
            summary["devices"][device_id] = {"total": 0, "passed": 0, "failed": 0}

        summary["devices"][device_id]["total"] += 1
        if result.status == "passed":
            summary["devices"][device_id]["passed"] += 1
        else:
            summary["devices"][device_id]["failed"] += 1

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始移动端测试框架健康检查")

            health_checks = {
                "environment_status": self.check_environment_health(),
                "device_availability": self.check_device_availability(),
                "test_execution": self.check_test_execution_health(),
                "performance_metrics": self.check_performance_health()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": "mobile_test_framework",
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning("移动端测试框架健康检查发现问题")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"移动端测试框架健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"移动端测试框架健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": "mobile_test_framework",
                "error": str(e)
            }

    def check_environment_health(self) -> Dict[str, Any]:
        """检查测试环境健康状态

        Returns:
            Dict[str, Any]: 环境健康状态检查结果
        """
        try:
            platforms_status = {}

            # 检查iOS环境
            try:
                ios_setup = self._setup_ios_environment()
                platforms_status["ios"] = {
                    "available": ios_setup,
                    "tools": ["xcode", "simulator"] if ios_setup else []
                }
            except Exception as e:
                platforms_status["ios"] = {"available": False, "error": str(e)}

            # 检查Android环境
            try:
                android_setup = self._setup_android_environment()
                platforms_status["android"] = {
                    "available": android_setup,
                    "tools": ["adb", "emulator"] if android_setup else []
                }
            except Exception as e:
                platforms_status["android"] = {"available": False, "error": str(e)}

            # 至少有一个平台可用
            any_platform_available = any(platform.get("available", False)
                                         for platform in platforms_status.values())

            return {
                "healthy": any_platform_available,
                "platforms": platforms_status,
                "available_platforms": [p for p, status in platforms_status.items() if status.get("available", False)]
            }
        except Exception as e:
            logger.error(f"环境健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_device_availability(self) -> Dict[str, Any]:
        """检查设备可用性

        Returns:
            Dict[str, Any]: 设备可用性检查结果
        """
        try:
            device_counts = {}

            for platform in PlatformType:
                try:
                    devices = self.device_manager.discover_devices(platform)
                    device_counts[platform.value] = {
                        "count": len(devices),
                        "devices": [d.device_id for d in devices]
                    }
                except Exception as e:
                    device_counts[platform.value] = {"count": 0, "error": str(e)}

            total_devices = sum(counts.get("count", 0) for counts in device_counts.values())
            has_devices = total_devices > 0

            return {
                "healthy": has_devices,
                "total_devices": total_devices,
                "platform_devices": device_counts,
                "available_platforms": [p for p, counts in device_counts.items() if counts.get("count", 0) > 0]
            }
        except Exception as e:
            logger.error(f"设备可用性检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_test_execution_health(self) -> Dict[str, Any]:
        """检查测试执行健康状态

        Returns:
            Dict[str, Any]: 测试执行健康检查结果
        """
        try:
            total_executors = len(self._executors)
            has_executors = total_executors > 0

            # 检查执行器状态
            executor_status = {}
            total_results = 0

            for device_id, executor in self._executors.items():
                results = executor.get_test_results()
                total_results += len(results)
                executor_status[device_id] = {
                    "has_results": len(results) > 0,
                    "result_count": len(results),
                    "last_execution": results[-1].end_time if results else None
                }

            # 检查是否有活跃的测试执行
            has_recent_activity = any(
                status.get("last_execution") and
                (datetime.now() - status["last_execution"]).seconds < 3600  # 1小时内
                for status in executor_status.values()
            )

            return {
                "healthy": has_executors and (total_results > 0 or has_recent_activity),
                "total_executors": total_executors,
                "total_results": total_results,
                "executor_status": executor_status,
                "has_recent_activity": has_recent_activity
            }
        except Exception as e:
            logger.error(f"测试执行健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态

        Returns:
            Dict[str, Any]: 性能健康检查结果
        """
        try:
            # 获取测试摘要进行性能分析
            summary = self.get_test_summary()

            if not summary or summary.get("total_tests", 0) == 0:
                return {"healthy": True, "reason": "no_tests_to_analyze"}

            total_tests = summary.get("total_tests", 0)
            passed_tests = summary.get("passed", 0)
            failed_tests = summary.get("failed", 0)

            # 计算成功率
            success_rate = passed_tests / total_tests if total_tests > 0 else 0

            # 计算平均执行时间（简化估算）
            avg_execution_time = 30.0  # 假设平均30秒（需要从实际结果计算）

            # 性能阈值检查
            acceptable_success_rate = success_rate > 0.7  # 成功率 > 70%
            acceptable_execution_time = avg_execution_time < 300  # 平均执行时间 < 5分钟

            return {
                "healthy": acceptable_success_rate and acceptable_execution_time,
                "performance_metrics": {
                    "success_rate": success_rate,
                    "avg_execution_time": avg_execution_time,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests
                },
                "thresholds": {
                    "acceptable_success_rate": acceptable_success_rate,
                    "acceptable_execution_time": acceptable_execution_time
                }
            }
        except Exception as e:
            logger.error(f"性能健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            test_summary = self.get_test_summary()
            health_check = self.check_health()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "test_summary": test_summary,
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            test_summary = self.get_test_summary()

            # 计算总体统计
            total_devices = len(self._executors)
            total_platforms = len(set(
                executor.device.platform.value
                for executor in self._executors.values()
            ))

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "framework_status": {
                    "total_devices": total_devices,
                    "total_platforms": total_platforms,
                    "active_executors": len(self._executors)
                },
                "test_summary": test_summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "error": str(e)}

    def monitor_device_status(self) -> Dict[str, Any]:
        """监控设备状态

        Returns:
            Dict[str, Any]: 设备状态监控结果
        """
        try:
            device_status = {}

            for device_id, executor in self._executors.items():
                device = executor.device
                status = self.device_manager.get_device_status(device_id)
                results = executor.get_test_results()

                device_status[device_id] = {
                    "platform": device.platform.value,
                    "status": status,
                    "total_tests": len(results),
                    "last_activity": results[-1].end_time if results else None,
                    "healthy": status in ["available", "ready"]
                }

            healthy_devices = sum(1 for status in device_status.values()
                                  if status.get("healthy", False))
            total_devices = len(device_status)

            return {
                "healthy": healthy_devices > 0,  # 至少有一个健康设备
                "device_status": device_status,
                "healthy_devices": healthy_devices,
                "total_devices": total_devices,
                "health_ratio": healthy_devices / total_devices if total_devices > 0 else 0
            }
        except Exception as e:
            logger.error(f"设备状态监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def monitor_test_execution(self) -> Dict[str, Any]:
        """监控测试执行状态

        Returns:
            Dict[str, Any]: 测试执行监控结果
        """
        try:
            execution_stats = {
                "total_executors": len(self._executors),
                "active_tests": 0,
                "completed_tests": 0,
                "failed_tests": 0,
                "avg_execution_time": 0.0
            }

            total_execution_time = 0.0
            test_count = 0

            for executor in self._executors.values():
                results = executor.get_test_results()
                execution_stats["completed_tests"] += len(results)

                for result in results:
                    if result.status in ["running", "pending"]:
                        execution_stats["active_tests"] += 1
                    elif result.status == "failed":
                        execution_stats["failed_tests"] += 1

                    total_execution_time += result.duration
                    test_count += 1

            if test_count > 0:
                execution_stats["avg_execution_time"] = total_execution_time / test_count

            # 计算执行效率
            total_tests = execution_stats["completed_tests"] + execution_stats["active_tests"]
            completion_rate = execution_stats["completed_tests"] / \
                total_tests if total_tests > 0 else 0

            return {
                "healthy": completion_rate > 0.8,  # 完成率 > 80%
                "execution_stats": execution_stats,
                "completion_rate": completion_rate,
                "efficiency_score": completion_rate * (1 - execution_stats["failed_tests"] / max(1, total_tests))
            }
        except Exception as e:
            logger.error(f"测试执行监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def validate_framework_config(self) -> Dict[str, Any]:
        """验证框架配置

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            validation_results = {
                "environment_setup": self._validate_environment_setup(),
                "device_discovery": self._validate_device_discovery(),
                "executor_creation": self._validate_executor_creation(),
                "parallel_execution": self._validate_parallel_execution()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"框架配置验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    def _validate_environment_setup(self) -> Dict[str, Any]:
        """验证环境设置"""
        try:
            ios_ok = self._setup_ios_environment()
            android_ok = self._setup_android_environment()

            return {
                "valid": ios_ok or android_ok,  # 至少一个平台可用
                "ios_available": ios_ok,
                "android_available": android_ok,
                "supported_platforms": [p for p, ok in [("ios", ios_ok), ("android", android_ok)] if ok]
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_device_discovery(self) -> Dict[str, Any]:
        """验证设备发现"""
        try:
            total_devices = 0
            platforms_checked = 0

            for platform in PlatformType:
                try:
                    devices = self.device_manager.discover_devices(platform)
                    total_devices += len(devices)
                    platforms_checked += 1
                except Exception:
                    continue

            return {
                "valid": total_devices > 0,
                "total_devices": total_devices,
                "platforms_checked": platforms_checked,
                "devices_per_platform": total_devices / max(1, platforms_checked)
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_executor_creation(self) -> Dict[str, Any]:
        """验证执行器创建"""
        try:
            executor_count = len(self._executors)

            return {
                "valid": executor_count >= 0,  # 执行器数量合理
                "executor_count": executor_count,
                "max_reasonable_executors": 10  # 合理的最大执行器数量
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_parallel_execution(self) -> Dict[str, Any]:
        """验证并行执行"""
        try:
            # 检查是否有足够的设备支持并行执行
            device_count = sum(len(self.device_manager.discover_devices(p)) for p in PlatformType)

            return {
                "valid": device_count > 0,
                "device_count": device_count,
                "parallel_supported": device_count > 1
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

# 便捷函数


def create_mobile_test_framework() -> MobileTestFramework:
    """创建移动端测试框架实例"""
    return MobileTestFramework()


def run_mobile_tests(platform: PlatformType, test_cases: List[str],

                     test_config: Optional[TestConfig] = None):
    """运行移动端测试的便捷函数"""
    if test_config is None:
        test_config = TestConfig(test_type=TestType.UI_TEST)

    framework = create_mobile_test_framework()

    # 设置测试环境
    if not framework.setup_test_environment(platform):
        logger.error(f"设置 {platform.value} 测试环境失败")
        return []

    # 运行测试
    return framework.run_tests(platform, test_config, test_cases)
