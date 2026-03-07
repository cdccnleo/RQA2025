#!/usr/bin/env python3
"""
检查PostgreSQL + TimescaleDB依赖和配置
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查Python依赖"""
    print("=" * 60)
    print("1. 检查Python依赖")
    print("=" * 60)
    
    dependencies = {
        "psycopg2": "psycopg2-binary",
        "sqlalchemy": "sqlalchemy",
    }
    
    all_ok = True
    for module_name, package_name in dependencies.items():
        try:
            if module_name == "psycopg2":
                import psycopg2
                version = psycopg2.__version__
            elif module_name == "sqlalchemy":
                import sqlalchemy
                version = sqlalchemy.__version__
            
            print(f"   ✅ {package_name}: {version}")
        except ImportError:
            print(f"   ❌ {package_name}: 未安装")
            print(f"      安装命令: pip install {package_name}")
            all_ok = False
    
    return all_ok


def check_environment_variables():
    """检查环境变量配置"""
    print("\n" + "=" * 60)
    print("2. 检查环境变量配置")
    print("=" * 60)
    
    env_vars = {
        "DB_HOST": os.getenv("DB_HOST", "未设置（默认: localhost）"),
        "DB_PORT": os.getenv("DB_PORT", "未设置（默认: 5432）"),
        "DB_NAME": os.getenv("DB_NAME", "未设置（默认: rqa2025）"),
        "DB_USER": os.getenv("DB_USER", "未设置（默认: rqa_user）"),
        "DB_PASSWORD": "已设置" if os.getenv("DB_PASSWORD") else "未设置（空密码）",
    }
    
    for key, value in env_vars.items():
        if key == "DB_PASSWORD":
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    # 检查是否有配置
    has_config = any([
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
        os.getenv("DB_USER"),
        os.getenv("DB_PASSWORD")
    ])
    
    if not has_config:
        print("\n   ⚠️  未设置环境变量，将使用默认配置")
        print("   建议设置以下环境变量:")
        print("   export DB_HOST=localhost")
        print("   export DB_PORT=5432")
        print("   export DB_NAME=rqa2025")
        print("   export DB_USER=rqa_user")
        print("   export DB_PASSWORD=your_password")
    
    return True


def check_database_config():
    """检查数据库配置"""
    print("\n" + "=" * 60)
    print("3. 检查数据库配置")
    print("=" * 60)
    
    try:
        from src.infrastructure.utils.components.environment import get_database_config
        config = get_database_config()
        
        print(f"   主机: {config.get('host', 'N/A')}")
        print(f"   端口: {config.get('port', 'N/A')}")
        print(f"   数据库: {config.get('name', 'N/A')}")
        print(f"   用户: {config.get('user', 'N/A')}")
        print(f"   SSL模式: {config.get('ssl_mode', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"   ❌ 获取配置失败: {e}")
        return False


def check_postgresql_connection():
    """检查PostgreSQL连接"""
    print("\n" + "=" * 60)
    print("4. 检查PostgreSQL连接")
    print("=" * 60)
    
    try:
        import psycopg2
        
        # 获取配置
        try:
            from src.infrastructure.utils.components.environment import get_database_config
            config = get_database_config()
        except:
            config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "rqa2025"),
                "user": os.getenv("DB_USER", "rqa_user"),
                "password": os.getenv("DB_PASSWORD", ""),
            }
        
        print(f"   尝试连接: {config['host']}:{config['port']}/{config['name']}")
        
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database=config["name"],
            user=config["user"],
            password=config["password"],
            connect_timeout=5
        )
        
        cursor = conn.cursor()
        
        # 检查PostgreSQL版本
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        print(f"   ✅ PostgreSQL连接成功")
        print(f"   版本: {pg_version.split(',')[0]}")
        
        # 检查TimescaleDB扩展
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
        has_timescaledb = cursor.fetchone()[0]
        
        if has_timescaledb:
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            ts_version = cursor.fetchone()[0]
            print(f"   ✅ TimescaleDB扩展已安装")
            print(f"   版本: {ts_version}")
        else:
            print(f"   ⚠️  TimescaleDB扩展未安装")
            print(f"   提示: 可以使用标准PostgreSQL表，功能不受影响")
            print(f"   安装TimescaleDB: https://docs.timescale.com/install/latest/")
        
        # 检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'akshare_stock_data'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print(f"   ✅ akshare_stock_data表已存在")
            
            # 检查是否是TimescaleDB超表
            if has_timescaledb:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM _timescaledb_catalog.hypertable 
                        WHERE hypertable_name = 'akshare_stock_data'
                    );
                """)
                is_hypertable = cursor.fetchone()[0]
                
                if is_hypertable:
                    print(f"   ✅ 表已转换为TimescaleDB超表")
                else:
                    print(f"   ⚠️  表存在但未转换为超表")
                    print(f"   提示: 运行初始化脚本可以自动转换")
        else:
            print(f"   ⚠️  akshare_stock_data表不存在")
            print(f"   提示: 运行 python scripts/init_akshare_database.py 创建表")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"   ❌ PostgreSQL连接失败")
        print(f"   错误: {str(e)}")
        print(f"\n   可能的原因:")
        print(f"   1. PostgreSQL服务未运行")
        print(f"   2. 数据库配置不正确")
        print(f"   3. 网络连接问题")
        print(f"   4. 用户权限不足")
        return False
    except ImportError:
        print(f"   ❌ psycopg2未安装，无法测试连接")
        return False
    except Exception as e:
        print(f"   ❌ 连接测试失败: {e}")
        return False


def check_persistence_module():
    """检查持久化模块"""
    print("\n" + "=" * 60)
    print("5. 检查持久化模块")
    print("=" * 60)
    
    try:
        from src.gateway.web import postgresql_persistence
        
        print(f"   ✅ postgresql_persistence模块可导入")
        
        # 检查关键函数
        functions = [
            "get_db_config",
            "get_db_connection",
            "ensure_table_exists",
            "persist_akshare_data_to_postgresql"
        ]
        
        for func_name in functions:
            if hasattr(postgresql_persistence, func_name):
                print(f"   ✅ 函数 {func_name} 存在")
            else:
                print(f"   ❌ 函数 {func_name} 不存在")
                return False
        
        return True
    except ImportError as e:
        print(f"   ❌ 模块导入失败: {e}")
        return False


def check_sql_files():
    """检查SQL文件"""
    print("\n" + "=" * 60)
    print("6. 检查SQL文件")
    print("=" * 60)
    
    sql_file = project_root / "scripts" / "sql" / "akshare_stock_data_schema.sql"
    
    if sql_file.exists():
        print(f"   ✅ SQL文件存在: {sql_file}")
        
        # 检查文件内容
        with open(sql_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "CREATE TABLE": "CREATE TABLE" in content,
            "UNIQUE约束": "UNIQUE" in content or "CONSTRAINT unique" in content.lower(),
            "索引": "CREATE INDEX" in content,
            "TimescaleDB": "timescaledb" in content.lower() or "create_hypertable" in content.lower(),
        }
        
        for check_name, result in checks.items():
            if result:
                print(f"   ✅ 包含 {check_name}")
            else:
                print(f"   ⚠️  缺少 {check_name}")
        
        return True
    else:
        print(f"   ❌ SQL文件不存在: {sql_file}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("PostgreSQL + TimescaleDB 依赖和配置检查")
    print("=" * 60)
    
    results = {
        "依赖": check_dependencies(),
        "环境变量": check_environment_variables(),
        "数据库配置": check_database_config(),
        "PostgreSQL连接": check_postgresql_connection(),
        "持久化模块": check_persistence_module(),
        "SQL文件": check_sql_files(),
    }
    
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！PostgreSQL + TimescaleDB配置正常")
    else:
        print("⚠️  部分检查未通过，请根据上述提示进行修复")
        print("\n建议操作:")
        print("1. 安装缺失的依赖: pip install -r requirements.txt")
        print("2. 设置环境变量（如需要）")
        print("3. 启动PostgreSQL服务")
        print("4. 运行初始化脚本: python scripts/init_akshare_database.py")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

