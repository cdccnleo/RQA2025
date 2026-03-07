#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试PostgreSQL连接修复"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_postgresql_connection():
    """测试PostgreSQL连接"""
    print("=" * 60)
    print("测试PostgreSQL连接修复...")
    print("=" * 60)
    
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection, get_db_config
        
        # 获取配置
        config = get_db_config()
        print(f"数据库配置:")
        print(f"  - 主机: {config['host']}")
        print(f"  - 端口: {config['port']}")
        print(f"  - 数据库: {config['database']}")
        print(f"  - 用户: {config['user']}")
        print(f"  - 密码: {'已设置' if config.get('password') else '未设置'}")
        
        # 尝试获取连接
        print("\n尝试连接PostgreSQL...")
        conn = get_db_connection()
        
        if conn:
            print("✅ PostgreSQL连接成功！")
            
            # 测试查询
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"✅ PostgreSQL版本: {version[0]}")
                cursor.close()
            except Exception as e:
                print(f"⚠️ 查询测试失败: {e}")
            
            # 归还连接
            return_db_connection(conn)
            return True
        else:
            print("❌ PostgreSQL连接失败（返回None）")
            print("   这可能是正常的，如果PostgreSQL服务未运行或配置不正确")
            print("   系统将自动回退到文件系统存储")
            return False
            
    except Exception as e:
        print(f"❌ PostgreSQL连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n注意：连接失败不影响系统功能，系统会自动使用文件系统存储")
        return False


if __name__ == '__main__':
    success = test_postgresql_connection()
    print("\n" + "=" * 60)
    if success:
        print("✅ PostgreSQL连接测试通过")
    else:
        print("⚠️ PostgreSQL连接测试失败（但不影响文件系统存储）")
    print("=" * 60)
    sys.exit(0 if success else 1)

