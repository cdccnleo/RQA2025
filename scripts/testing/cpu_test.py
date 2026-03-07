#!/usr/bin/env python3
"""
CPU计算能力测试脚本
验证CPU版本的PyTorch是否正常工作
"""

import torch
import time
import numpy as np

def test_cpu_basic():
    """基础CPU测试"""
    print("🔍 基础CPU测试")
    print("="*50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"设备类型: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"CPU线程数: {torch.get_num_threads()}")
    
    if not torch.cuda.is_available():
        print("✅ 使用CPU版本PyTorch")
        return True
    else:
        print("❌ 检测到CUDA，应该使用CPU版本")
        return False

def test_cpu_computation():
    """CPU计算测试"""
    print("\n🧮 CPU计算测试")
    print("="*50)
    
    # 创建测试数据
    size = 1000
    print(f"创建 {size}x{size} 矩阵...")
    
    # CPU计算测试
    start_time = time.time()
    
    # 矩阵乘法测试
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    c = torch.mm(a, b)
    
    cpu_time = time.time() - start_time
    print(f"CPU矩阵乘法耗时: {cpu_time:.4f}秒")
    
    # 验证结果
    result_sum = torch.sum(c).item()
    print(f"结果矩阵元素和: {result_sum:.2f}")
    
    return cpu_time < 10.0  # 10秒内完成算正常

def test_cpu_memory():
    """CPU内存测试"""
    print("\n💾 CPU内存测试")
    print("="*50)
    
    try:
        # 测试大矩阵创建
        size = 5000
        print(f"创建 {size}x{size} 大矩阵...")
        
        start_time = time.time()
        large_matrix = torch.randn(size, size)
        creation_time = time.time() - start_time
        
        memory_usage = large_matrix.numel() * 4 / (1024**2)  # MB
        print(f"矩阵大小: {memory_usage:.2f} MB")
        print(f"创建耗时: {creation_time:.4f}秒")
        
        # 清理内存
        del large_matrix
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False

def test_cpu_performance():
    """CPU性能测试"""
    print("\n⚡ CPU性能测试")
    print("="*50)
    
    # 多次计算取平均值
    times = []
    for i in range(5):
        start_time = time.time()
        
        # 执行一些计算密集型操作
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        c = torch.mm(a, b)
        d = torch.mm(c, a)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"第{i+1}次计算耗时: {elapsed:.4f}秒")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"平均耗时: {avg_time:.4f}秒 (±{std_time:.4f})")
    
    return avg_time < 5.0  # 平均5秒内完成算正常

def main():
    """主函数"""
    print("🚀 CPU计算能力测试开始")
    print("="*60)
    
    results = []
    
    # 基础测试
    results.append(("基础测试", test_cpu_basic()))
    
    # 计算测试
    results.append(("计算测试", test_cpu_computation()))
    
    # 内存测试
    results.append(("内存测试", test_cpu_memory()))
    
    # 性能测试
    results.append(("性能测试", test_cpu_performance()))
    
    # 结果汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 CPU版本PyTorch工作正常！")
        return True
    else:
        print("⚠️ CPU版本PyTorch存在问题，需要检查")
        return False

if __name__ == "__main__":
    main() 