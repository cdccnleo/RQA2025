#!/usr/bin/env python3
"""
GPU功能测试脚本
验证GPU计算能力和性能
"""

import torch
import time
import numpy as np

def test_gpu_basic():
    """基础GPU测试"""
    print("🔍 基础GPU测试")
    print("="*50)
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 测试GPU内存
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU已用内存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU缓存内存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

def test_gpu_computation():
    """GPU计算性能测试"""
    print("\n🧮 GPU计算性能测试")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过计算测试")
        return False
    
    device = torch.device('cuda:0')
    
    # 测试1: 小矩阵乘法
    print("测试1: 小矩阵乘法 (1000x1000)")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
    
    # 测试2: 大矩阵乘法
    print("测试2: 大矩阵乘法 (5000x5000)")
    x = torch.randn(5000, 5000).to(device)
    y = torch.randn(5000, 5000).to(device)
    
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
    
    # 测试3: 批量操作
    print("测试3: 批量卷积操作")
    batch_size = 32
    channels = 64
    height = 224
    width = 224
    
    input_tensor = torch.randn(batch_size, channels, height, width).to(device)
    conv_layer = torch.nn.Conv2d(channels, 128, kernel_size=3, padding=1).to(device)
    
    start_time = time.time()
    output = conv_layer(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
    print(f"输出形状: {output.shape}")
    
    return True

def test_gpu_memory():
    """GPU内存测试"""
    print("\n💾 GPU内存测试")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存测试")
        return False
    
    device = torch.device('cuda:0')
    
    # 获取GPU内存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    
    print(f"总内存: {total_memory / 1024**3:.2f} GB")
    print(f"已分配: {allocated_memory / 1024**3:.2f} GB")
    print(f"已预留: {reserved_memory / 1024**3:.2f} GB")
    print(f"使用率: {allocated_memory / total_memory * 100:.1f}%")
    
    # 测试内存分配
    print("\n测试内存分配...")
    try:
        # 尝试分配大块内存
        large_tensor = torch.randn(1000, 1000, 1000).to(device)
        print(f"✅ 成功分配大张量: {large_tensor.numel() * 4 / 1024**3:.2f} GB")
        
        # 清理内存
        del large_tensor
        torch.cuda.empty_cache()
        print("✅ 内存清理成功")
        
    except RuntimeError as e:
        print(f"❌ 内存分配失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 GPU功能测试开始")
    print("="*60)
    
    # 基础测试
    if not test_gpu_basic():
        print("❌ 基础GPU测试失败")
        return
    
    # 计算性能测试
    if not test_gpu_computation():
        print("❌ GPU计算测试失败")
        return
    
    # 内存测试
    if not test_gpu_memory():
        print("❌ GPU内存测试失败")
        return
    
    print("\n" + "="*60)
    print("🎉 所有GPU测试通过！")
    print("✅ GPU功能正常，可以用于深度学习任务")
    print("="*60)

if __name__ == "__main__":
    main() 