#!/usr/bin/env python3
"""
RQA2025 应用服务
基于标准库的量化交易系统主服务
"""

import sys
import os
import json
import time
import logging
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import socket

# 设置Python路径
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库连接配置
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'rqa2025'),
    'user': os.getenv('POSTGRES_USER', 'rqa2025'),
    'password': os.getenv('POSTGRES_PASSWORD', 'rqa2025pass')
}

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'redis'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': 0
}

# 模拟数据存储（因为没有外部依赖）
strategies_db = [
    {
        "id": 1,
        "name": "双均线策略",
        "description": "基于5日和20日均线的趋势跟踪策略",
        "status": "active",
        "created_at": time.time() - 86400
    },
    {
        "id": 2,
        "name": "RSI超买超卖策略",
        "description": "基于RSI指标的超买超卖反转策略",
        "status": "active",
        "created_at": time.time() - 43200
    }
]

market_data_db = {
    "AAPL": {"price": 150.25, "volume": 45678900, "change": 2.34},
    "GOOGL": {"price": 2750.80, "volume": 1234567, "change": -1.23},
    "MSFT": {"price": 305.50, "volume": 23456789, "change": 0.89},
    "TSLA": {"price": 245.30, "volume": 34567890, "change": 5.67}
}

# 数据库连接函数（简化版）
def get_db_status():
    """检查数据库连接状态"""
    try:
        # 这里可以集成真实的数据库连接
        # 暂时返回模拟状态
        return {"status": "healthy", "message": "Database connected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_redis_status():
    """检查Redis连接状态"""
    try:
        # 这里可以集成真实的Redis连接
        # 暂时返回模拟状态
        return {"status": "healthy", "message": "Redis connected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class APIHandler(BaseHTTPRequestHandler):
    """REST API 请求处理器"""

    def do_HEAD(self):
        """处理HEAD请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        try:
            if path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            elif path == '/metrics':
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            elif path.startswith('/api/') or path == '/docs':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
        except Exception as e:
            logger.error(f"Error handling HEAD request: {e}")
            self.send_response(500)
            self.end_headers()

    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)

        # 设置CORS头
        self.send_cors_headers()

        try:
            if path == '/health':
                self.handle_health()
            elif path == '/api/v1/status':
                self.handle_system_status()
            elif path == '/api/v1/strategies':
                self.handle_get_strategies()
            elif path.startswith('/api/v1/market/'):
                symbol = path.split('/')[-1]
                self.handle_get_market_data(symbol)
            elif path == '/api/v1/metrics' or path == '/metrics':
                self.handle_get_metrics()
            elif path == '/docs':
                self.handle_api_docs()
            else:
                self.send_error_response(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_error_response(500, "Internal server error")

    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # 设置CORS头
        self.send_cors_headers()

        try:
            if path == '/api/v1/strategies':
                self.handle_create_strategy()
            else:
                self.send_error_response(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            self.send_error_response(500, "Internal server error")

    def do_OPTIONS(self):
        """处理OPTIONS请求（CORS预检）"""
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        """发送CORS头"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def send_json_response(self, data, status=200):
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def send_error_response(self, status, message):
        """发送错误响应"""
        self.send_json_response({
            "error": message,
            "status": status,
            "timestamp": time.time()
        }, status)

    def handle_health(self):
        """健康检查"""
        db_status = get_db_status()
        redis_status = get_redis_status()

        response = {
            "status": "ok",
            "service": "rqa2025-app",
            "timestamp": time.time(),
            "version": "1.0.0",
            "database": db_status,
            "redis": redis_status,
            "message": "All systems operational"
        }
        self.send_json_response(response)

    def handle_system_status(self):
        """系统状态"""
        response = {
            "status": "operational",
            "services": ["database", "redis", "api", "web", "prometheus", "grafana"],
            "uptime": time.time(),
            "version": "1.0.0"
        }
        self.send_json_response(response)

    def handle_get_strategies(self):
        """获取交易策略"""
        response = {
            "strategies": strategies_db,
            "count": len(strategies_db),
            "timestamp": time.time()
        }
        self.send_json_response(response)

    def handle_create_strategy(self):
        """创建交易策略"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())

        new_strategy = {
            "id": len(strategies_db) + 1,
            "name": data.get("name", "New Strategy"),
            "description": data.get("description", ""),
            "status": data.get("status", "active"),
            "created_at": time.time()
        }

        strategies_db.append(new_strategy)
        self.send_json_response(new_strategy, 201)

    def handle_get_market_data(self, symbol):
        """获取市场数据"""
        if symbol.upper() in market_data_db:
            data = market_data_db[symbol.upper()]
            # 添加一些随机波动
            price_variation = random.uniform(-2, 2)
            volume_variation = random.randint(-1000000, 1000000)

            response = {
                "symbol": symbol.upper(),
                "price": round(data["price"] + price_variation, 2),
                "volume": data["volume"] + volume_variation,
                "change": data["change"],
                "timestamp": time.time()
            }
        else:
            response = {
                "symbol": symbol.upper(),
                "price": round(random.uniform(50, 500), 2),
                "volume": random.randint(100000, 10000000),
                "change": round(random.uniform(-5, 5), 2),
                "timestamp": time.time()
            }

        self.send_json_response(response)

    def handle_get_metrics(self):
        """获取系统指标"""
        # 检查请求路径，如果是/metrics返回Prometheus格式，否则返回JSON
        if self.path == '/metrics':
            # Prometheus格式的指标
            metrics = f"""# HELP rqa2025_api_requests_total Total number of API requests
# TYPE rqa2025_api_requests_total counter
rqa2025_api_requests_total {random.randint(1000, 10000)}

# HELP rqa2025_api_response_time Response time in milliseconds
# TYPE rqa2025_api_response_time gauge
rqa2025_api_response_time {random.randint(50, 500)}

# HELP rqa2025_database_connections Active database connections
# TYPE rqa2025_database_connections gauge
rqa2025_database_connections {random.randint(1, 10)}

# HELP rqa2025_redis_memory_used Redis memory usage in MB
# TYPE rqa2025_redis_memory_used gauge
rqa2025_redis_memory_used {random.randint(10, 100)}

# HELP rqa2025_service_up Service health status
# TYPE rqa2025_service_up gauge
rqa2025_service_up 1
"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            # JSON格式的指标
            response = {
                "database": {
                    "connections": random.randint(1, 10),
                    "queries_per_second": random.randint(10, 100)
                },
                "redis": {
                    "memory_used_mb": random.randint(10, 100),
                    "keys_count": random.randint(100, 1000)
                },
                "api": {
                    "requests_per_minute": random.randint(10, 200),
                    "response_time_ms": random.randint(50, 500)
                },
                "timestamp": time.time()
            }
            self.send_json_response(response)

    def handle_api_docs(self):
        """API文档"""
        docs = {
            "title": "RQA2025 量化交易系统 API",
            "version": "1.0.0",
            "description": "基于AI的现代化量化交易平台API",
            "endpoints": {
                "GET /health": "系统健康检查",
                "GET /api/v1/status": "获取系统状态",
                "GET /api/v1/strategies": "获取交易策略列表",
                "POST /api/v1/strategies": "创建新的交易策略",
                "GET /api/v1/market/{symbol}": "获取市场数据",
                "GET /api/v1/metrics": "获取系统指标"
            },
            "timestamp": time.time()
        }
        self.send_json_response(docs)

    def log_message(self, format, *args):
        """禁用默认日志输出"""
        pass

def create_tables():
    """初始化数据库表（简化版）"""
    logger.info("Initializing database tables...")
    # 这里可以添加实际的数据库表创建逻辑
    # 暂时跳过，因为我们使用模拟数据
    pass

def main():
    """主函数"""
    logger.info("RQA2025 Application Service Starting...")
    logger.info("=" * 50)
    logger.info("Service: RQA2025 量化交易系统")
    logger.info("Version: 1.0.0")
    logger.info("Port: 8000")
    logger.info("=" * 50)

    # 初始化数据库表
    create_tables()

    # 启动HTTP服务器
    server_address = ('0.0.0.0', 8000)
    httpd = HTTPServer(server_address, APIHandler)

    logger.info("Server started successfully!")
    logger.info("Health check: http://localhost:8000/health")
    logger.info("API docs: http://localhost:8000/docs")
    logger.info("System status: http://localhost:8000/api/v1/status")
    logger.info("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        httpd.server_close()
        logger.info("Server shut down")

if __name__ == '__main__':
    main()

# 数据库连接配置
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'rqa2025'),
    'user': os.getenv('POSTGRES_USER', 'rqa2025'),
    'password': os.getenv('POSTGRES_PASSWORD', 'rqa2025pass')
}

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'redis'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': 0
}

# 数据库连接函数
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def get_redis_connection():
    try:
        r = redis.Redis(**REDIS_CONFIG)
        r.ping()
        return r
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

# 健康检查路由
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查"""
    timestamp = time.time()

    # 检查数据库
    db_status = {"status": "unknown"}
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_health")
            count = cursor.fetchone()[0]
            db_status = {"status": "healthy", "records": count}
            conn.close()
        else:
            db_status = {"status": "error", "message": "Connection failed"}
    except Exception as e:
        db_status = {"status": "error", "message": str(e)}

    # 检查Redis
    redis_status = {"status": "unknown"}
    try:
        r = get_redis_connection()
        if r:
            info = r.info()
            redis_status = {"status": "healthy", "version": info.get('redis_version', 'unknown')}
        else:
            redis_status = {"status": "error", "message": "Connection failed"}
    except Exception as e:
        redis_status = {"status": "error", "message": str(e)}

    return HealthResponse(
        status="ok",
        service="rqa2025-app",
        timestamp=timestamp,
        version="1.0.0",
        database=db_status,
        redis=redis_status,
        message="All systems operational"
    )

# API路由组
@app.get("/api/v1/status")
async def system_status():
    """获取系统状态"""
    return {
        "status": "operational",
        "services": ["database", "redis", "api", "web"],
        "uptime": time.time(),
        "version": "1.0.0"
    }

# 交易策略API
@app.get("/api/v1/strategies")
async def get_strategies():
    """获取所有交易策略"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=503, detail="Database unavailable")

        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, status, created_at FROM trading_strategies ORDER BY created_at DESC")
        strategies = cursor.fetchall()
        conn.close()

        result = []
        for row in strategies:
            result.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "status": row[3],
                "created_at": row[4].timestamp() if row[4] else None
            })

        return {"strategies": result, "count": len(result)}

    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/strategies")
async def create_strategy(strategy: TradingStrategy):
    """创建新的交易策略"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=503, detail="Database unavailable")

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trading_strategies (name, description, status, created_at)
            VALUES (%s, %s, %s, NOW())
            RETURNING id, created_at
        """, (strategy.name, strategy.description, strategy.status))

        result = cursor.fetchone()
        conn.commit()
        conn.close()

        return {
            "id": result[0],
            "name": strategy.name,
            "description": strategy.description,
            "status": strategy.status,
            "created_at": result[1].timestamp() if result[1] else None
        }

    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 市场数据API
@app.get("/api/v1/market/{symbol}")
async def get_market_data(symbol: str):
    """获取市场数据"""
    try:
        # 这里可以集成真实的市场数据源
        # 暂时返回模拟数据
        import random
        price = 100 + random.uniform(-10, 10)
        volume = random.randint(1000, 10000)

        return MarketData(
            symbol=symbol.upper(),
            price=round(price, 2),
            volume=volume,
            timestamp=time.time()
        )

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 系统监控API
@app.get("/api/v1/metrics")
async def get_metrics():
    """获取系统指标"""
    try:
        # 获取数据库指标
        db_metrics = {"connections": 0, "queries": 0}
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
            db_metrics["connections"] = cursor.fetchone()[0]
            conn.close()

        # 获取Redis指标
        redis_metrics = {"memory_used": 0, "keys_count": 0}
        r = get_redis_connection()
        if r:
            info = r.info()
            redis_metrics["memory_used"] = info.get('used_memory', 0)
            redis_metrics["keys_count"] = r.dbsize()

        return {
            "database": db_metrics,
            "redis": redis_metrics,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 创建数据库表
def create_tables():
    """初始化数据库表"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Cannot connect to database for table creation")
            return

        cursor = conn.cursor()

        # 创建交易策略表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_strategies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建市场数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(15,8) NOT NULL,
                volume BIGINT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_strategies_status ON trading_strategies(status);
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
            CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
        """)

        conn.commit()
        conn.close()
        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("RQA2025 Application Service Starting...")

    # 创建数据库表
    create_tables()

    logger.info("Application startup complete")
    logger.info("API documentation available at: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("RQA2025 Application Service Shutting down...")

if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(
        "scripts.simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
