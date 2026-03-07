#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成器
为持续优化引擎生成模拟使用数据
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class UsageData:
    """使用数据"""
    timestamp: float
    user_id: str
    operation_type: str
    response_time: float
    success: bool
    error_message: str = None
    performance_metrics: Dict[str, float] = None


class DataGenerator:
    """数据生成器"""

    def __init__(self):
        self.operation_types = [
            "cache_get", "cache_put", "risk_check",
            "parameter_optimization", "monitoring_check",
            "data_analysis", "model_training", "report_generation"
        ]

        self.base_response_times = {
            "cache_get": 15.0,
            "cache_put": 25.0,
            "risk_check": 45.0,
            "parameter_optimization": 120.0,
            "monitoring_check": 30.0,
            "data_analysis": 80.0,
            "model_training": 300.0,
            "report_generation": 60.0
        }

        self.success_rates = {
            "cache_get": 0.98,
            "cache_put": 0.95,
            "risk_check": 0.92,
            "parameter_optimization": 0.88,
            "monitoring_check": 0.96,
            "data_analysis": 0.90,
            "model_training": 0.85,
            "report_generation": 0.93
        }

    def generate_usage_data(self, num_points: int = 200, hours_back: int = 24) -> List[UsageData]:
        """生成使用数据"""
        print(f"📊 生成 {num_points} 个使用数据点...")

        current_time = time.time()
        data_points = []

        # 生成时间分布（更多数据在最近时间）
        time_distribution = self._generate_time_distribution(num_points, hours_back)

        for i in range(num_points):
            # 生成时间戳
            timestamp = current_time - time_distribution[i]

            # 选择操作类型
            operation_type = random.choice(self.operation_types)

            # 生成响应时间
            base_time = self.base_response_times[operation_type]
            response_time = base_time + random.uniform(-base_time * 0.2, base_time * 0.3)

            # 生成成功率
            success_rate = self.success_rates[operation_type]
            success = random.random() < success_rate

            # 生成性能指标
            performance_metrics = self._generate_performance_metrics(operation_type)

            # 生成用户ID
            user_id = f"user_{random.randint(1, 15)}"

            # 创建数据点
            usage_data = UsageData(
                timestamp=timestamp,
                user_id=user_id,
                operation_type=operation_type,
                response_time=response_time,
                success=success,
                error_message=None if success else f"{operation_type}操作失败",
                performance_metrics=performance_metrics
            )

            data_points.append(usage_data)

        return data_points

    def _generate_time_distribution(self, num_points: int, hours_back: int) -> List[float]:
        """生成时间分布"""
        # 使用指数分布，更多数据在最近时间
        times = []
        for _ in range(num_points):
            # 指数分布，偏向最近时间
            time_offset = random.expovariate(1.0 / (hours_back * 3600 * 0.3))
            time_offset = min(time_offset, hours_back * 3600)
            times.append(time_offset)

        return sorted(times, reverse=True)  # 从最近到最远

    def _generate_performance_metrics(self, operation_type: str) -> Dict[str, float]:
        """生成性能指标"""
        metrics = {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(40, 85),
            "cache_hit_rate": random.uniform(0.5, 0.98),
            "error_rate": random.uniform(0, 0.15),
            "network_latency": random.uniform(5, 50)
        }

        # 根据操作类型调整指标
        if "cache" in operation_type:
            metrics["cache_hit_rate"] = random.uniform(0.6, 0.98)
        elif "training" in operation_type:
            metrics["cpu_usage"] = random.uniform(60, 95)
            metrics["memory_usage"] = random.uniform(70, 90)
        elif "analysis" in operation_type:
            metrics["cpu_usage"] = random.uniform(40, 75)

        return metrics

    def save_data_to_file(self, data_points: List[UsageData], filename: str = "usage_data.json") -> Path:
        """保存数据到文件"""
        output_dir = Path("data/continuous_optimization/")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转换为字典格式
        data_dict = {
            "generated_at": time.time(),
            "total_points": len(data_points),
            "data_points": [asdict(point) for point in data_points]
        }

        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)

        print(f"📄 数据已保存到: {file_path}")
        return file_path

    def load_data_from_file(self, filename: str = "usage_data.json") -> List[UsageData]:
        """从文件加载数据"""
        file_path = Path("data/continuous_optimization/") / filename

        if not file_path.exists():
            print(f"⚠️ 文件不存在: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        # 转换回UsageData对象
        data_points = []
        for point_dict in data_dict["data_points"]:
            data_point = UsageData(
                timestamp=point_dict["timestamp"],
                user_id=point_dict["user_id"],
                operation_type=point_dict["operation_type"],
                response_time=point_dict["response_time"],
                success=point_dict["success"],
                error_message=point_dict.get("error_message"),
                performance_metrics=point_dict.get("performance_metrics", {})
            )
            data_points.append(data_point)

        print(f"📊 从文件加载了 {len(data_points)} 个数据点")
        return data_points


def main():
    """主函数"""
    print("📊 启动数据生成器...")

    # 创建数据生成器
    generator = DataGenerator()

    # 生成使用数据
    data_points = generator.generate_usage_data(num_points=300, hours_back=48)

    # 保存数据
    file_path = generator.save_data_to_file(data_points, "usage_data.json")

    # 统计信息
    print("\n" + "="*50)
    print("📊 数据生成统计:")
    print("="*50)

    # 按操作类型统计
    operation_counts = {}
    success_counts = {}
    response_times = {}

    for point in data_points:
        op_type = point.operation_type
        if op_type not in operation_counts:
            operation_counts[op_type] = 0
            success_counts[op_type] = 0
            response_times[op_type] = []

        operation_counts[op_type] += 1
        if point.success:
            success_counts[op_type] += 1
        response_times[op_type].append(point.response_time)

    print(f"总数据点: {len(data_points)}")
    print(f"时间范围: 过去48小时")
    print(f"操作类型分布:")

    for op_type, count in operation_counts.items():
        success_rate = success_counts[op_type] / count
        avg_response_time = sum(response_times[op_type]) / len(response_times[op_type])
        print(f"  {op_type}: {count} 次, 成功率: {success_rate:.2%}, 平均响应时间: {avg_response_time:.1f}ms")

    # 整体统计
    total_success = sum(1 for point in data_points if point.success)
    overall_success_rate = total_success / len(data_points)
    avg_response_time = sum(point.response_time for point in data_points) / len(data_points)

    print(f"\n整体统计:")
    print(f"  总成功率: {overall_success_rate:.2%}")
    print(f"  平均响应时间: {avg_response_time:.1f}ms")

    print("="*50)
    print("✅ 数据生成完成!")


if __name__ == "__main__":
    main()
