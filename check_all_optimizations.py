#!/usr/bin/env python3
"""
全面检查所有优化实现

检查清单:
1. 股票池管理器
2. 批量任务API
3. 增量计算机制
4. 并行计算优化
5. 前端界面优化
"""

import os
import sys
import importlib
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class OptimizationChecker:
    """优化检查器"""
    
    def __init__(self):
        self.check_results = []
        self.errors = []
    
    def check_file_exists(self, filepath, description):
        """检查文件是否存在"""
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        self.check_results.append({
            "item": f"文件存在: {description}",
            "status": status,
            "detail": filepath if exists else f"文件不存在: {filepath}"
        })
        return exists
    
    def check_module_import(self, module_path, description):
        """检查模块是否可以导入"""
        try:
            module = importlib.import_module(module_path)
            self.check_results.append({
                "item": f"模块导入: {description}",
                "status": "✅",
                "detail": f"成功导入 {module_path}"
            })
            return True, module
        except Exception as e:
            self.check_results.append({
                "item": f"模块导入: {description}",
                "status": "❌",
                "detail": str(e)
            })
            self.errors.append(f"导入 {module_path} 失败: {e}")
            return False, None
    
    def check_class_exists(self, module, class_name, description):
        """检查类是否存在"""
        exists = hasattr(module, class_name)
        status = "✅" if exists else "❌"
        self.check_results.append({
            "item": f"类存在: {description}",
            "status": status,
            "detail": f"{class_name} 存在" if exists else f"{class_name} 不存在"
        })
        return exists
    
    def check_function_exists(self, module, func_name, description):
        """检查函数是否存在"""
        exists = hasattr(module, func_name) and callable(getattr(module, func_name))
        status = "✅" if exists else "❌"
        self.check_results.append({
            "item": f"函数存在: {description}",
            "status": status,
            "detail": f"{func_name} 存在" if exists else f"{func_name} 不存在"
        })
        return exists
    
    def run_all_checks(self):
        """运行所有检查"""
        print("=" * 80)
        print("A股市场大规模特征计算架构优化 - 全面检查")
        print("=" * 80)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 检查股票池管理器
        print("\n" + "=" * 80)
        print("1. 股票池管理器 (StockPoolManager)")
        print("=" * 80)
        self._check_stock_pool_manager()
        
        # 2. 检查批量任务API
        print("\n" + "=" * 80)
        print("2. 批量任务API")
        print("=" * 80)
        self._check_batch_api()
        
        # 3. 检查增量计算机制
        print("\n" + "=" * 80)
        print("3. 增量计算机制 (IncrementalCalculator)")
        print("=" * 80)
        self._check_incremental_calculator()
        
        # 4. 检查并行计算优化
        print("\n" + "=" * 80)
        print("4. 并行计算优化 (ParallelCalculator)")
        print("=" * 80)
        self._check_parallel_calculator()
        
        # 5. 检查前端界面
        print("\n" + "=" * 80)
        print("5. 前端界面优化")
        print("=" * 80)
        self._check_frontend()
        
        # 6. 生成最终报告
        self._generate_report()
    
    def _check_stock_pool_manager(self):
        """检查股票池管理器"""
        # 检查文件
        filepath = "src/data_management/stock_pool_manager.py"
        self.check_file_exists(filepath, "股票池管理器文件")
        
        # 检查模块导入
        success, module = self.check_module_import(
            "src.data_management.stock_pool_manager",
            "股票池管理器模块"
        )
        
        if success:
            # 检查类
            self.check_class_exists(module, "StockPoolManager", "StockPoolManager类")
            self.check_class_exists(module, "StockPool", "StockPool类")
            self.check_class_exists(module, "StockInfo", "StockInfo类")
            self.check_class_exists(module, "StockPoolType", "StockPoolType枚举")
            
            # 检查函数
            self.check_function_exists(module, "get_stock_pool_manager", "get_stock_pool_manager函数")
            self.check_function_exists(module, "close_stock_pool_manager", "close_stock_pool_manager函数")
            
            # 检查关键方法
            if hasattr(module, 'StockPoolManager'):
                manager_class = getattr(module, 'StockPoolManager')
                methods = ['get_all_stocks', 'get_predefined_pool', 'split_pool_for_batch', 
                          'get_pool_statistics', 'create_custom_pool']
                for method in methods:
                    has_method = hasattr(manager_class, method)
                    status = "✅" if has_method else "❌"
                    self.check_results.append({
                        "item": f"方法存在: StockPoolManager.{method}",
                        "status": status,
                        "detail": f"方法存在" if has_method else f"方法不存在"
                    })
    
    def _check_batch_api(self):
        """检查批量任务API"""
        # 检查路由文件
        filepath = "src/gateway/web/feature_engineering_routes.py"
        self.check_file_exists(filepath, "特征工程路由文件")
        
        # 检查模块导入
        success, module = self.check_module_import(
            "src.gateway.web.feature_engineering_routes",
            "特征工程路由模块"
        )
        
        if success:
            # 检查关键函数
            self.check_function_exists(module, "create_batch_feature_tasks_endpoint", "批量任务创建端点")
            self.check_function_exists(module, "get_stock_pools_endpoint", "股票池列表端点")
    
    def _check_incremental_calculator(self):
        """检查增量计算机制"""
        # 检查文件
        filepath = "src/features/core/incremental_calculator.py"
        self.check_file_exists(filepath, "增量计算器文件")
        
        # 检查模块导入
        success, module = self.check_module_import(
            "src.features.core.incremental_calculator",
            "增量计算器模块"
        )
        
        if success:
            # 检查类
            self.check_class_exists(module, "IncrementalFeatureCalculator", "IncrementalFeatureCalculator类")
            self.check_class_exists(module, "CalculationRecord", "CalculationRecord类")
            self.check_class_exists(module, "IncrementalConfig", "IncrementalConfig类")
            
            # 检查函数
            self.check_function_exists(module, "get_incremental_calculator", "get_incremental_calculator函数")
            
            # 检查关键方法
            if hasattr(module, 'IncrementalFeatureCalculator'):
                calc_class = getattr(module, 'IncrementalFeatureCalculator')
                methods = ['needs_recalculation', 'calculate_incremental', 'batch_calculate_incremental',
                          'get_cached_features', 'cache_features', 'merge_features']
                for method in methods:
                    has_method = hasattr(calc_class, method)
                    status = "✅" if has_method else "❌"
                    self.check_results.append({
                        "item": f"方法存在: IncrementalFeatureCalculator.{method}",
                        "status": status,
                        "detail": f"方法存在" if has_method else f"方法不存在"
                    })
    
    def _check_parallel_calculator(self):
        """检查并行计算优化"""
        # 检查文件
        filepath = "src/features/core/parallel_calculator.py"
        self.check_file_exists(filepath, "并行计算器文件")
        
        # 检查模块导入
        success, module = self.check_module_import(
            "src.features.core.parallel_calculator",
            "并行计算器模块"
        )
        
        if success:
            # 检查类
            self.check_class_exists(module, "ParallelFeatureCalculator", "ParallelFeatureCalculator类")
            self.check_class_exists(module, "ParallelConfig", "ParallelConfig类")
            self.check_class_exists(module, "ParallelResult", "ParallelResult类")
            self.check_class_exists(module, "BatchProgress", "BatchProgress类")
            
            # 检查函数
            self.check_function_exists(module, "get_parallel_calculator", "get_parallel_calculator函数")
            self.check_function_exists(module, "benchmark_parallel_calculation", "benchmark_parallel_calculation函数")
            
            # 检查关键方法
            if hasattr(module, 'ParallelFeatureCalculator'):
                calc_class = getattr(module, 'ParallelFeatureCalculator')
                methods = ['calculate_batch', 'calculate_batch_async', 'calculate_with_async_io',
                          'get_progress', 'shutdown']
                for method in methods:
                    has_method = hasattr(calc_class, method)
                    status = "✅" if has_method else "❌"
                    self.check_results.append({
                        "item": f"方法存在: ParallelFeatureCalculator.{method}",
                        "status": status,
                        "detail": f"方法存在" if has_method else f"方法不存在"
                    })
    
    def _check_frontend(self):
        """检查前端界面"""
        # 检查HTML文件
        filepath = "web-static/feature-engineering-monitor.html"
        exists = self.check_file_exists(filepath, "特征工程监控页面")
        
        if exists:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查关键元素
            checks = [
                ("批量任务模式选择", 'name="taskMode"'),
                ("股票池选择器", 'id="stockPoolSelect"'),
                ("单只股票输入", 'id="singleSymbol"'),
                ("日期选择器", 'id="startDate"'),
                ("技术指标多选", 'name="indicators"'),
                ("批量大小配置", 'id="batchSize"'),
                ("任务模式切换函数", 'toggleTaskMode'),
                ("股票池信息加载", 'loadStockPoolInfo'),
            ]
            
            for name, pattern in checks:
                found = pattern in content
                status = "✅" if found else "❌"
                self.check_results.append({
                    "item": f"前端元素: {name}",
                    "status": status,
                    "detail": f"找到 '{pattern}'" if found else f"未找到 '{pattern}'"
                })
    
    def _generate_report(self):
        """生成最终报告"""
        print("\n" + "=" * 80)
        print("检查报告汇总")
        print("=" * 80)
        
        # 统计结果
        total = len(self.check_results)
        passed = sum(1 for r in self.check_results if r["status"] == "✅")
        failed = sum(1 for r in self.check_results if r["status"] == "❌")
        
        print(f"\n总检查项: {total}")
        print(f"通过: {passed} ✅")
        print(f"失败: {failed} ❌")
        print(f"通过率: {passed/total*100:.1f}%" if total > 0 else "N/A")
        
        # 显示详细结果
        print("\n" + "-" * 80)
        print("详细检查结果:")
        print("-" * 80)
        
        for i, result in enumerate(self.check_results, 1):
            print(f"{i:3d}. {result['status']} {result['item']}")
            if result['status'] == "❌":
                print(f"     详情: {result['detail']}")
        
        # 显示错误汇总
        if self.errors:
            print("\n" + "-" * 80)
            print("错误汇总:")
            print("-" * 80)
            for error in self.errors:
                print(f"❌ {error}")
        
        # 最终结论
        print("\n" + "=" * 80)
        if failed == 0:
            print("🎉 所有检查通过！A股市场大规模特征计算架构优化已全部完成！")
        elif failed <= 3:
            print("⚠️ 大部分检查通过，有少量问题需要修复。")
        else:
            print("❌ 有多项检查失败，需要进一步修复。")
        print("=" * 80)
        
        # 保存报告到文件
        report_file = "optimization_check_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("A股市场大规模特征计算架构优化 - 检查报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总检查项: {total}\n")
            f.write(f"通过: {passed}\n")
            f.write(f"失败: {failed}\n")
            f.write(f"通过率: {passed/total*100:.1f}%\n" if total > 0 else "N/A\n")
            f.write("\n详细结果:\n")
            f.write("-" * 80 + "\n")
            for i, result in enumerate(self.check_results, 1):
                f.write(f"{i:3d}. {result['status']} {result['item']}\n")
                if result['status'] == "❌":
                    f.write(f"     详情: {result['detail']}\n")
        
        print(f"\n详细报告已保存到: {report_file}")


if __name__ == "__main__":
    checker = OptimizationChecker()
    checker.run_all_checks()
