"""
仪表盘数据获取验证脚本
检查每个仪表盘的数据获取情况并生成报告
"""

import requests
import json
from typing import Dict, Any, List
from datetime import datetime


BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"
WEB_BASE = "http://localhost:8080"


class DashboardVerifier:
    """仪表盘验证器"""
    
    def __init__(self):
        self.results = []
    
    def verify_api(self, name: str, endpoint: str, expected_keys: List[str] = None):
        """验证API端点"""
        try:
            response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
            status = response.status_code
            
            if status == 200:
                try:
                    data = response.json()
                    has_data = bool(data)
                    has_expected_keys = True
                    
                    if expected_keys and isinstance(data, dict):
                        has_expected_keys = all(key in data for key in expected_keys)
                    
                    result = {
                        "name": name,
                        "endpoint": endpoint,
                        "status": "✅ 正常",
                        "http_code": status,
                        "has_data": has_data,
                        "has_expected_keys": has_expected_keys,
                        "data_type": type(data).__name__
                    }
                except json.JSONDecodeError:
                    result = {
                        "name": name,
                        "endpoint": endpoint,
                        "status": "⚠️ 响应非JSON格式",
                        "http_code": status,
                        "has_data": False,
                        "has_expected_keys": False,
                        "data_type": "text"
                    }
            elif status == 404:
                result = {
                    "name": name,
                    "endpoint": endpoint,
                    "status": "❌ 端点不存在",
                    "http_code": status,
                    "has_data": False,
                    "has_expected_keys": False,
                    "data_type": None
                }
            else:
                result = {
                    "name": name,
                    "endpoint": endpoint,
                    "status": f"⚠️ HTTP {status}",
                    "http_code": status,
                    "has_data": False,
                    "has_expected_keys": False,
                    "data_type": None
                }
        except requests.exceptions.RequestException as e:
            result = {
                "name": name,
                "endpoint": endpoint,
                "status": f"❌ 连接失败: {str(e)[:50]}",
                "http_code": None,
                "has_data": False,
                "has_expected_keys": False,
                "data_type": None
            }
        
        self.results.append(result)
        return result
    
    def verify_page(self, name: str, path: str):
        """验证页面加载"""
        try:
            response = requests.get(f"{WEB_BASE}{path}", timeout=5)
            status = response.status_code
            
            if status == 200:
                content_length = len(response.text)
                has_content = content_length > 1000
                
                result = {
                    "name": name,
                    "path": path,
                    "status": "✅ 正常" if has_content else "⚠️ 内容异常",
                    "http_code": status,
                    "content_length": content_length
                }
            else:
                result = {
                    "name": name,
                    "path": path,
                    "status": f"❌ HTTP {status}",
                    "http_code": status,
                    "content_length": 0
                }
        except requests.exceptions.RequestException as e:
            result = {
                "name": name,
                "path": path,
                "status": f"❌ 连接失败: {str(e)[:50]}",
                "http_code": None,
                "content_length": 0
            }
        
        self.results.append(result)
        return result
    
    def generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 80)
        print("  RQA2025 仪表盘数据获取验证报告")
        print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        
        # API端点验证
        print("## API端点验证\n")
        api_results = [r for r in self.results if "endpoint" in r]
        
        for result in api_results:
            print(f"{result['status']} {result['name']}")
            print(f"  端点: {result['endpoint']}")
            print(f"  HTTP状态: {result['http_code']}")
            if result.get('has_data'):
                print(f"  数据: 有数据 ({result.get('data_type', 'unknown')})")
            else:
                print(f"  数据: 无数据")
            print()
        
        # 页面验证
        print("## 页面加载验证\n")
        page_results = [r for r in self.results if "path" in r]
        
        for result in page_results:
            print(f"{result['status']} {result['name']}")
            print(f"  路径: {result['path']}")
            print(f"  HTTP状态: {result['http_code']}")
            print(f"  内容长度: {result.get('content_length', 0)} 字节")
            print()
        
        # 统计汇总
        print("## 验证统计\n")
        total = len(self.results)
        success = sum(1 for r in self.results if "✅" in r.get('status', ''))
        warning = sum(1 for r in self.results if "⚠️" in r.get('status', ''))
        failed = sum(1 for r in self.results if "❌" in r.get('status', ''))
        
        print(f"总计: {total}")
        print(f"正常: {success} ✅")
        print(f"警告: {warning} ⚠️")
        print(f"失败: {failed} ❌")
        print(f"成功率: {success/total*100:.1f}%")
        
        # 问题识别
        if failed > 0 or warning > 0:
            print("\n## 问题识别\n")
            
            failed_results = [r for r in self.results if "❌" in r.get('status', '')]
            if failed_results:
                print("### 失败项:\n")
                for result in failed_results:
                    print(f"- {result['name']}: {result.get('status', '未知错误')}")
            
            warning_results = [r for r in self.results if "⚠️" in r.get('status', '')]
            if warning_results:
                print("\n### 警告项:\n")
                for result in warning_results:
                    print(f"- {result['name']}: {result.get('status', '未知警告')}")


def main():
    """主验证流程"""
    verifier = DashboardVerifier()
    
    # Phase 1: 量化策略开发流程
    print("Phase 1: 量化策略开发流程验证...")
    
    # 1.1 数据收集阶段
    verifier.verify_api("数据源列表", "/data/sources", ["sources"])
    verifier.verify_api("数据源指标", "/data-sources/metrics")
    verifier.verify_api("数据质量指标", "/data/quality/metrics")
    
    # 1.2 特征工程监控
    verifier.verify_api("特征任务列表", "/features/engineering/tasks", ["tasks"])
    verifier.verify_api("特征列表", "/features/engineering/features", ["features"])
    verifier.verify_api("技术指标状态", "/features/engineering/indicators", ["indicators"])
    
    # 1.3 模型训练监控
    verifier.verify_api("训练任务列表", "/ml/training/jobs", ["jobs"])
    verifier.verify_api("训练指标", "/ml/training/metrics")
    
    # 1.4 策略性能评估
    verifier.verify_api("策略对比", "/strategy/performance/comparison", ["strategies"])
    verifier.verify_api("策略性能指标", "/strategy/performance/metrics")
    
    # Phase 2: 交易执行流程
    print("Phase 2: 交易执行流程验证...")
    
    # 2.2 交易信号生成监控
    verifier.verify_api("实时信号", "/trading/signals/realtime", ["signals"])
    verifier.verify_api("信号统计", "/trading/signals/stats", ["stats"])
    verifier.verify_api("信号分布", "/trading/signals/distribution")
    
    # 2.3 订单智能路由监控
    verifier.verify_api("路由决策", "/trading/routing/decisions", ["decisions"])
    verifier.verify_api("路由统计", "/trading/routing/stats", ["stats"])
    verifier.verify_api("路由性能", "/trading/routing/performance")
    
    # Phase 3: 风险控制流程
    print("Phase 3: 风险控制流程验证...")
    
    # 3.1 风险报告生成
    verifier.verify_api("报告模板", "/risk/reporting/templates", ["templates"])
    verifier.verify_api("报告任务", "/risk/reporting/tasks", ["tasks"])
    verifier.verify_api("报告历史", "/risk/reporting/history", ["reports"])
    verifier.verify_api("报告统计", "/risk/reporting/stats", ["stats"])
    
    # 页面验证
    print("页面加载验证...")
    verifier.verify_page("特征工程监控", "/feature-engineering-monitor")
    verifier.verify_page("模型训练监控", "/model-training-monitor")
    verifier.verify_page("策略性能评估", "/strategy-performance-evaluation")
    verifier.verify_page("交易信号监控", "/trading-signal-monitor")
    verifier.verify_page("订单路由监控", "/order-routing-monitor")
    verifier.verify_page("风险报告生成", "/risk-reporting")
    
    # 生成报告
    verifier.generate_report()


if __name__ == "__main__":
    main()

