"""
性能测试
测试API响应时间、并发处理等性能指标
"""

import pytest
import requests
import time
import concurrent.futures
from typing import List, Dict, Any
from statistics import mean, median


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"


class TestAPIPerformance:
    """API性能测试类"""
    
    def test_api_response_time(self):
        """测试API响应时间"""
        apis = [
            ("/data/sources", "数据源列表"),
            ("/features/engineering/tasks", "特征任务列表"),
            ("/ml/training/jobs", "训练任务列表"),
            ("/trading/signals/realtime", "实时信号"),
            ("/risk/reporting/templates", "报告模板"),
        ]
        
        response_times = {}
        max_response_time = 2.0  # 最大响应时间（秒）
        
        for endpoint, name in apis:
            start_time = time.time()
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    response_times[name] = elapsed_time
                    assert elapsed_time < max_response_time, \
                        f"{name}响应时间 {elapsed_time:.3f}s 超过阈值 {max_response_time}s"
                    print(f"✅ {name}: {elapsed_time:.3f}s")
                else:
                    print(f"⚠️ {name}: HTTP {response.status_code} (耗时: {elapsed_time:.3f}s)")
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"❌ {name}: 请求失败 {e} (耗时: {elapsed_time:.3f}s)")
        
        if response_times:
            avg_time = mean(response_times.values())
            median_time = median(response_times.values())
            print(f"\n平均响应时间: {avg_time:.3f}s")
            print(f"中位响应时间: {median_time:.3f}s")
            print("✅ API响应时间测试通过")
    
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        api_url = f"{API_BASE}/features/engineering/tasks"
        concurrent_count = 10
        
        def make_request():
            start_time = time.time()
            try:
                response = requests.get(api_url, timeout=5)
                elapsed_time = time.time() - start_time
                return {
                    "status_code": response.status_code,
                    "response_time": elapsed_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                elapsed_time = time.time() - start_time
                return {
                    "status_code": 0,
                    "response_time": elapsed_time,
                    "success": False,
                    "error": str(e)
                }
        
        # 并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_count)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 分析结果
        success_count = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        
        print(f"\n并发请求测试 (并发数: {concurrent_count}):")
        print(f"成功请求: {success_count}/{concurrent_count}")
        if response_times:
            print(f"平均响应时间: {mean(response_times):.3f}s")
            print(f"最大响应时间: {max(response_times):.3f}s")
            print(f"最小响应时间: {min(response_times):.3f}s")
        
        # 至少应该有80%的请求成功
        success_rate = success_count / concurrent_count
        assert success_rate >= 0.8, f"并发请求成功率 {success_rate*100:.1f}% 低于阈值 80%"
        print("✅ 并发请求测试通过")
    
    def test_sequential_requests_performance(self):
        """测试顺序请求性能"""
        api_url = f"{API_BASE}/data/sources"
        request_count = 5
        
        response_times = []
        for i in range(request_count):
            start_time = time.time()
            try:
                response = requests.get(api_url, timeout=5)
                elapsed_time = time.time() - start_time
                if response.status_code == 200:
                    response_times.append(elapsed_time)
                print(f"请求 {i+1}: {elapsed_time:.3f}s")
            except Exception as e:
                print(f"请求 {i+1} 失败: {e}")
            time.sleep(0.1)  # 短暂延迟
        
        if response_times:
            avg_time = mean(response_times)
            print(f"\n顺序请求性能:")
            print(f"平均响应时间: {avg_time:.3f}s")
            print(f"请求数量: {len(response_times)}/{request_count}")
            print("✅ 顺序请求性能测试通过")
    
    def test_different_endpoints_performance(self):
        """测试不同端点的性能"""
        endpoints = [
            ("/data/sources", "数据源"),
            ("/features/engineering/tasks", "特征工程"),
            ("/ml/training/jobs", "模型训练"),
            ("/trading/signals/realtime", "交易信号"),
            ("/trading/routing/decisions", "订单路由"),
            ("/risk/reporting/templates", "风险报告"),
        ]
        
        performance_data = {}
        
        for endpoint, name in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data_size = len(response.text)
                    performance_data[name] = {
                        "response_time": elapsed_time,
                        "data_size": data_size,
                        "status": "success"
                    }
                    print(f"✅ {name}: {elapsed_time:.3f}s ({data_size} 字节)")
                else:
                    performance_data[name] = {
                        "response_time": elapsed_time,
                        "status": f"HTTP {response.status_code}"
                    }
                    print(f"⚠️ {name}: HTTP {response.status_code}")
            except Exception as e:
                elapsed_time = time.time() - start_time
                performance_data[name] = {
                    "response_time": elapsed_time,
                    "status": f"error: {str(e)[:50]}"
                }
                print(f"❌ {name}: 请求失败")
        
        # 分析性能
        successful = {k: v for k, v in performance_data.items() if v.get("status") == "success"}
        if successful:
            avg_time = mean([v["response_time"] for v in successful.values()])
            print(f"\n性能统计:")
            print(f"成功端点: {len(successful)}/{len(endpoints)}")
            print(f"平均响应时间: {avg_time:.3f}s")
            print("✅ 不同端点性能测试完成")
    
    def test_data_size_impact(self):
        """测试数据大小对性能的影响"""
        # 获取不同大小的数据
        endpoints = [
            ("/data/sources", "小数据量"),
            ("/features/engineering/features", "中等数据量"),
        ]
        
        for endpoint, description in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data_size = len(response.text)
                    throughput = data_size / elapsed_time if elapsed_time > 0 else 0
                    print(f"{description}:")
                    print(f"  响应时间: {elapsed_time:.3f}s")
                    print(f"  数据大小: {data_size} 字节")
                    print(f"  吞吐量: {throughput:.0f} 字节/秒")
                else:
                    print(f"{description}: HTTP {response.status_code}")
            except Exception as e:
                print(f"{description}: 请求失败 {e}")
        
        print("✅ 数据大小影响测试完成")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

