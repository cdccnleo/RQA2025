#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python路径配置修复脚本
修复项目配置文件中的硬编码路径
"""

import json
import os
import sys
from pathlib import Path


def get_current_project_root():
    """获取当前项目根目录"""
    # 从脚本位置向上查找项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return str(project_root.absolute())


def backup_file(file_path):
    """备份文件"""
    if os.path.exists(file_path):
        backup_path = file_path + '.bak'
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份: {file_path} -> {backup_path}")
        return True
    return False


def update_json_config(config_file, current_path):
    """更新JSON配置文件中的路径"""
    try:
        if not os.path.exists(config_file):
            print(f"⚠️ 文件不存在: {config_file}")
            return False
        
        # 备份
        backup_file(config_file)
        
        # 读取
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新路径
        updated = False
        
        if 'project_root' in config:
            config['project_root'] = current_path
            updated = True
        
        if 'python_path' in config:
            config['python_path'] = current_path
            updated = True
        
        if 'working_directory' in config:
            config['working_directory'] = current_path
            updated = True
        
        if 'environment' in config:
            if 'PYTHONPATH' in config['environment']:
                config['environment']['PYTHONPATH'] = current_path
                updated = True
        
        if 'env_vars' in config:
            if 'PYTHONPATH' in config['env_vars']:
                config['env_vars']['PYTHONPATH'] = current_path
                updated = True
        
        if updated:
            # 保存
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已更新: {config_file}")
            return True
        else:
            print(f"ℹ️ 无需更新: {config_file}")
            return False
    
    except Exception as e:
        print(f"❌ 错误: {config_file} - {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Python路径配置修复脚本")
    print("=" * 60)
    
    # 获取当前项目根目录
    current_path = get_current_project_root()
    print(f"\n当前项目根目录: {current_path}")
    
    # 需要更新的配置文件列表
    config_files = [
        'test_environment_config.json',
        'config/service_config.json',
        'training_env/config/service_config.json',
    ]
    
    print(f"\n准备更新 {len(config_files)} 个配置文件...\n")
    
    # 更新配置文件
    updated_count = 0
    for config_file in config_files:
        if update_json_config(config_file, current_path):
            updated_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"✅ 完成！共更新了 {updated_count} 个配置文件")
    print("=" * 60)
    
    # 提示清理缓存
    print("\n⚠️ 重要：请运行以下命令清理Python缓存：")
    print("Windows PowerShell:")
    print("Get-ChildItem -Path . -Recurse -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force")
    print("\nBash:")
    print("find . -type d -name '__pycache__' -exec rm -rf {} +")


if __name__ == '__main__':
    main()

