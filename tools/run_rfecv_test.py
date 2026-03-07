#!/usr/bin/env python3
"""
RFECV测试专用运行脚本
包含超时控制和死锁检测，避免测试无限挂起
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def run_test_with_timeout(test_path, timeout_seconds=300):
    """
    运行测试并设置超时控制
    
    Args:
        test_path: 测试文件路径
        timeout_seconds: 超时时间（秒）
    """
    print(f"🚀 开始运行RFECV测试: {test_path}")
    print(f"⏰ 超时设置: {timeout_seconds}秒")
    
    # 构建pytest命令
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "--timeout=120",  # 使用pytest-timeout插件
        "-k", "test_rfecv_medium_dataset"  # 只运行有问题的测试
    ]
    
    print(f"📋 执行命令: {' '.join(cmd)}")
    
    try:
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        start_time = time.time()
        
        # 监控进程输出和超时
        while process.poll() is None:
            # 检查是否超时
            if time.time() - start_time > timeout_seconds:
                print(f"⏰ 测试超时（{timeout_seconds}秒），强制终止进程")
                process.terminate()
                
                # 等待进程结束
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️ 进程未响应，强制杀死")
                    process.kill()
                
                return False, f"测试超时（{timeout_seconds}秒）"
            
            # 检查输出
            if process.stdout:
                output = process.stdout.readline()
                if output:
                    print(f"📤 {output.strip()}")
            
            time.sleep(0.1)
        
        # 获取最终结果
        stdout, stderr = process.communicate()
        execution_time = time.time() - start_time
        
        print(f"⏱️ 测试执行时间: {execution_time:.2f}秒")
        
        if process.returncode == 0:
            print("✅ 测试成功完成")
            return True, stdout
        else:
            print(f"❌ 测试失败，返回码: {process.returncode}")
            if stderr:
                print(f"错误输出: {stderr}")
            return False, stderr
            
    except Exception as e:
        print(f"💥 运行测试时发生异常: {e}")
        return False, str(e)

def run_specific_test_method():
    """运行特定的测试方法"""
    test_file = "tests/unit/features/test_rfecv_performance.py"
    
    if not Path(test_file).exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    print("🔍 检查测试环境...")
    
    # 检查pytest-timeout插件
    try:
        import pytest_timeout
        print("✅ pytest-timeout插件已安装")
    except ImportError:
        print("⚠️ pytest-timeout插件未安装，尝试安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-timeout"], 
                         check=True, capture_output=True)
            print("✅ pytest-timeout插件安装成功")
        except subprocess.CalledProcessError:
            print("❌ pytest-timeout插件安装失败")
            return False
    
    # 运行测试
    success, output = run_test_with_timeout(test_file, timeout_seconds=300)
    
    if success:
        print("🎉 RFECV测试成功完成！")
        print("📊 测试输出:")
        print(output)
    else:
        print("💥 RFECV测试失败或超时")
        print("📋 错误信息:")
        print(output)
        
        # 提供解决建议
        print("\n🔧 解决建议:")
        print("1. 检查AdvancedFeatureSelector的RFECV实现")
        print("2. 减少数据集规模或特征数量")
        print("3. 降低交叉验证折数")
        print("4. 添加max_features参数限制")
        print("5. 检查是否存在死循环或死锁")
    
    return success

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 RFECV性能测试专用运行脚本")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📍 当前工作目录: {current_dir}")
    
    # 检查项目结构
    if not (current_dir / "src").exists():
        print("⚠️ 警告: 未在项目根目录运行，可能影响测试")
    
    # 运行测试
    success = run_specific_test_method()
    
    print("=" * 60)
    if success:
        print("🎯 测试执行完成")
    else:
        print("⚠️ 测试存在问题，请检查上述建议")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
