#!/usr/bin/env python3
"""
重启后端服务
停止占用8000端口的进程并重新启动后端服务
"""

import sys
import os
import subprocess
import time
import socket
from pathlib import Path

def kill_process_on_port(port=8000):
    """停止占用指定端口的进程"""
    try:
        # Windows系统使用netstat和taskkill
        if sys.platform == 'win32':
            # 查找占用端口的进程
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            
            pids = []
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids.append(pid)
            
            if pids:
                print(f"发现占用端口{port}的进程: {', '.join(set(pids))}")
                for pid in set(pids):
                    try:
                        print(f"正在停止进程 {pid}...")
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True, check=False)
                        print(f"✅ 进程 {pid} 已停止")
                    except Exception as e:
                        print(f"⚠️  停止进程 {pid} 失败: {e}")
                
                # 等待端口释放
                time.sleep(2)
                return True
            else:
                print(f"未发现占用端口{port}的进程")
                return False
        else:
            # Linux/Mac系统
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"发现占用端口{port}的进程: {', '.join(pids)}")
                for pid in pids:
                    try:
                        print(f"正在停止进程 {pid}...")
                        subprocess.run(['kill', '-9', pid], check=True)
                        print(f"✅ 进程 {pid} 已停止")
                    except Exception as e:
                        print(f"⚠️  停止进程 {pid} 失败: {e}")
                
                time.sleep(2)
                return True
            else:
                print(f"未发现占用端口{port}的进程")
                return False
                
    except Exception as e:
        print(f"检查端口占用时出错: {e}")
        return False

def check_port_available(port=8000, host='localhost'):
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0
    except Exception:
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("重启 RQA2025 后端服务")
    print("=" * 60)
    
    # 停止占用端口的进程
    print("\n1. 检查并停止占用端口8000的进程...")
    kill_process_on_port(8000)
    
    # 检查端口是否已释放
    print("\n2. 检查端口是否已释放...")
    if check_port_available(8000):
        print("   ✅ 端口8000已释放")
    else:
        print("   ⚠️  端口8000仍被占用，等待3秒后重试...")
        time.sleep(3)
        if not check_port_available(8000):
            print("   ❌ 端口8000仍被占用，请手动停止相关进程")
            return
    
    # 启动后端服务
    print("\n3. 启动后端服务...")
    print("   (使用 scripts/start_api_server.py 启动)")
    print("\n   请在新的终端窗口中运行:")
    print("   python scripts/start_api_server.py")
    print("\n   或者运行:")
    print("   python scripts/start_backend.py")

if __name__ == "__main__":
    main()
