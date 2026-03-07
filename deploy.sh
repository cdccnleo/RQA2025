#!/bin / bash"
# 生产环境部署脚本
# 数据库配置
export DB_HOST=db_host
    export DB_PORT=5432
        export DB_USER=user
            export DB_PASS=pass
                # 交易参数
export MAX_ORDER_SIZE=10000
    export DEFAULT_SLIPPAGE=0.001
        # 启动服务
./start_service.sh --env=prod
    