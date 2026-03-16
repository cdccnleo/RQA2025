

def save_features_to_store(task_id: str, features: List[str], symbol: str = None, 
                          feature_types: Dict[str, str] = None, 
                          quality_scores: Dict[str, float] = None) -> bool:
    """
    保存特征到特征存储表
    
    Args:
        task_id: 任务ID
        features: 特征名称列表
        symbol: 股票代码
        feature_types: 特征类型字典 {特征名: 类型}
        quality_scores: 特征质量评分字典 {特征名: 评分}
    
    Returns:
        是否成功保存
    """
    logger.info(f"💾 save_features_to_store 被调用，任务ID: {task_id}, 特征数量: {len(features)}, 股票代码: {symbol}")
    
    # 过滤基础价格特征（双重保障）
    basic_price_features = {'open', 'high', 'low', 'close', 'volume', 'amount', 'date', 'datetime', 'timestamp'}
    filtered_features = [f for f in features if f.lower() not in basic_price_features]
    filtered_count = len(features) - len(filtered_features)
    if filtered_count > 0:
        logger.info(f"📝 save_features_to_store 内部过滤了 {filtered_count} 个基础价格特征")
    
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.error("❌ PostgreSQL连接不可用，无法保存特征到存储表")
            return False
        
        logger.info(f"✅ PostgreSQL连接成功")
        
        # 确保feature_store表存在
        if not _ensure_feature_store_table(conn):
            logger.error("❌ 无法创建feature_store表，保存操作取消")
            return False
        
        cursor = conn.cursor()
        logger.info(f"📝 开始插入 {len(filtered_features)} 个特征到数据库")
        
        # 解析特征名称，提取特征类型和参数
        import re
        
        inserted_count = 0
        for feature_name in filtered_features:
            # 生成特征ID
            feature_id = f"{task_id}_{feature_name}"
            
            # 解析特征类型和参数
            feature_type = None
            parameters = {}
            
            if feature_types and feature_name in feature_types:
                feature_type = feature_types[feature_name]
            else:
                # 尝试从特征名解析，如 SMA_5, EMA_10
                match = re.match(r'([A-Za-z]+)_(\d+)', feature_name)
                if match:
                    feature_type = match.group(1).upper()
                    parameters['period'] = int(match.group(2))
                else:
                    # 尝试其他格式，如 RSI, MACD
                    base_name = feature_name.split('_')[0] if '_' in feature_name else feature_name
                    feature_type = base_name.upper()
            
            # 获取质量评分
            quality_score = quality_scores.get(feature_name) if quality_scores else None
            
            # 插入或更新特征记录
            # 使用 psycopg2 的 JSONB 适配，直接传递 Python 字典
            import psycopg2.extras
            cursor.execute("""
                INSERT INTO feature_store 
                (feature_id, task_id, feature_name, feature_type, parameters, symbol, quality_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_id) DO UPDATE SET
                    feature_type = EXCLUDED.feature_type,
                    parameters = EXCLUDED.parameters,
                    quality_score = EXCLUDED.quality_score,
                    updated_at = CURRENT_TIMESTAMP
            """, (feature_id, task_id, feature_name, feature_type, 
                  psycopg2.extras.Json(parameters) if parameters else None, 
                  symbol, quality_score))
            inserted_count += 1
        
        conn.commit()
        logger.info(f"✅ 数据库提交成功，插入/更新 {inserted_count} 个特征")
        cursor.close()
        return_db_connection(conn)
        
        logger.info(f"✅ 已保存 {len(features)} 个特征到特征存储表，任务ID: {task_id}")
        return True
        
    except Exception as e:
        logger.error(f"保存特征到存储表失败: {e}")
        return False


def get_features_from_store(task_id: str = None, symbol: str = None, 
                           feature_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    从特征存储表获取特征列表
    
    Args:
        task_id: 任务ID（可选）
        symbol: 股票代码（可选）
        feature_type: 特征类型（可选）
        limit: 返回数量限制
    
    Returns:
        特征列表
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法从存储表获取特征")
            return []
        
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if task_id:
            conditions.append("task_id = %s")
            params.append(task_id)
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        if feature_type:
            conditions.append("feature_type = %s")
            params.append(feature_type)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        cursor.execute(f"""
            SELECT feature_id, task_id, feature_name, feature_type, parameters, 
                   symbol, quality_score, importance, created_at, updated_at
            FROM feature_store
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """, params + [limit])
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        features = []
        for row in rows:
            features.append({
                "feature_id": row[0],
                "task_id": row[1],
                "name": row[2],
                "feature_type": row[3],
                "parameters": row[4] if row[4] else {},
                "symbol": row[5],
                "quality_score": row[6],
                "importance": row[7],
                "created_at": row[8].timestamp() if row[8] else None,
                "updated_at": row[9].timestamp() if row[9] else None
            })
        
        return features
        
    except Exception as e:
        logger.error(f"从存储表获取特征失败: {e}")
        return []


def delete_features_from_store_by_task(task_id: str) -> int:
    """
    根据任务ID删除feature_store表中的特征数据
    
    Args:
        task_id: 任务ID
    
    Returns:
        删除的特征数量
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法删除特征存储数据")
            return 0
        
        cursor = conn.cursor()
        
        # 先查询有多少条记录
        cursor.execute("SELECT COUNT(*) FROM feature_store WHERE task_id = %s", (task_id,))
        count = cursor.fetchone()[0]
        
        # 删除关联的特征数据
        cursor.execute("DELETE FROM feature_store WHERE task_id = %s", (task_id,))
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"已从feature_store删除 {count} 个特征数据，任务ID: {task_id}")
        return count
        
    except Exception as e:
        logger.error(f"从feature_store删除特征数据失败: {e}")
        return 0


def update_feature_task_status(task_id: str, status: str, feature_count: int = None, error_message: str = None) -> bool:
    """
    更新特征提取任务状态
    
    用于调度器在任务完成或失败时同步更新数据库状态
    
    Args:
        task_id: 任务ID
        status: 新状态 (completed/failed/running等)
        feature_count: 特征数量（可选）
        error_message: 错误信息（可选）
    
    Returns:
        是否成功更新
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法更新任务状态")
            return False
        
        cursor = conn.cursor()
        
        # 构建更新SQL
        update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if feature_count is not None:
            update_fields.append("feature_count = %s")
            params.append(feature_count)
        
        if error_message is not None:
            update_fields.append("error_message = %s")
            params.append(error_message)
        
        # 如果状态是completed或failed，设置end_time
        if status in ["completed", "failed"]:
            update_fields.append("end_time = %s")
            params.append(int(time.time()))
        
        # 添加task_id到参数列表
        params.append(task_id)
        
        # 执行更新
        sql = f"""
            UPDATE feature_engineering_tasks 
            SET {', '.join(update_fields)}
            WHERE task_id = %s
        """
        
        cursor.execute(sql, tuple(params))
        conn.commit()
        
        updated_rows = cursor.rowcount
        cursor.close()
        return_db_connection(conn)
        
        if updated_rows > 0:
            logger.debug(f"任务状态已更新到数据库: {task_id} -> {status}")
            return True
        else:
            logger.warning(f"数据库中没有找到任务: {task_id}")
            return False
            
    except Exception as e:
        logger.error(f"更新任务状态到数据库失败: {e}")
        return False


def get_features(symbol: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    获取特征列表
    
    用于特征选择处理器获取可用特征数据
    
    Args:
        symbol: 股票代码（可选，如果指定则只返回该股票的特征）
        limit: 返回数量限制
    
    Returns:
        特征列表，每个特征包含name, symbol, quality_score等字段
    """
    try:
        logger.info(f"🔍 获取特征列表: symbol={symbol}, limit={limit}")
        
        # 优先从PostgreSQL的feature_store表获取
        features = get_features_from_store(symbol=symbol, limit=limit)
        
        if features:
            logger.info(f"✅ 从feature_store获取到 {len(features)} 个特征")
            return features
        
        # 如果PostgreSQL没有数据，从文件系统获取
        logger.debug("PostgreSQL中没有特征数据，尝试从文件系统获取")
        
        # 从feature_tasks目录加载所有任务，提取特征信息
        all_features = []
        
        if os.path.exists(FEATURE_TASKS_DIR):
            for filename in os.listdir(FEATURE_TASKS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(FEATURE_TASKS_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            task = json.load(f)
                        
                        # 从任务配置中提取特征信息
                        config = task.get('config', {})
                        task_symbol = config.get('symbol', config.get('stock_code', ''))
                        
                        # 如果指定了symbol，过滤匹配的任务
                        if symbol and task_symbol != symbol:
                            continue
                        
                        # 从config中提取特征列表
                        indicators = config.get('indicators', [])
                        for indicator in indicators:
                            feature = {
                                'name': indicator,
                                'symbol': task_symbol,
                                'feature_type': 'technical',
                                'quality_score': 0.8,  # 默认质量分数
                                'task_id': task.get('task_id', ''),
                                'created_at': task.get('created_at', 0)
                            }
                            all_features.append(feature)
                            
                    except Exception as e:
                        logger.warning(f"读取任务文件失败 {filename}: {e}")
        
        if all_features:
            logger.info(f"✅ 从文件系统获取到 {len(all_features)} 个特征")
            return all_features[:limit]
        
        logger.warning("没有找到任何特征数据")
        return []
        
    except Exception as e:
        logger.error(f"获取特征列表失败: {e}")
        return []


def _ensure_feature_store_table(conn) -> bool:
    """
    确保feature_store表存在
    
    Args:
        conn: 数据库连接
    
    Returns:
        是否成功
    """
    try:
        cursor = conn.cursor()
        
        # 创建feature_store表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                feature_id VARCHAR(255) PRIMARY KEY,
                task_id VARCHAR(100) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50),
                parameters JSONB,
                symbol VARCHAR(20),
                quality_score FLOAT,
                importance FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_task_id 
            ON feature_store(task_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_symbol 
            ON feature_store(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_feature_type 
            ON feature_store(feature_type)
        """)
        
        conn.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"创建feature_store表失败: {e}")
        return False


# 别名函数，用于兼容其他模块的调用
def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取特征提取任务（load_feature_task的别名）
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务信息字典，不存在返回None
    """
    return load_feature_task(task_id)


def update_task_status(
    task_id: str,
    status: str,
    progress: int = None,
    result: Dict = None,
    error_message: str = None
) -> bool:
    """
    更新特征提取任务状态
    
    Args:
        task_id: 任务ID
        status: 新状态
        progress: 进度（0-100）
        result: 执行结果
        error_message: 错误信息
    
    Returns:
        是否成功更新
    """
    try:
        # 优先更新PostgreSQL
        pg_success = False
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]
                
                if progress is not None:
                    update_fields.append("progress = %s")
                    params.append(progress)
                
                if error_message is not None:
                    update_fields.append("error_message = %s")
                    params.append(error_message)
                
                # 如果状态是completed或failed，设置end_time
                if status in ["completed", "failed"]:
                    update_fields.append("end_time = %s")
                    params.append(int(time.time()))
                
                params.append(task_id)
                
                cursor.execute(f"""
                    UPDATE feature_engineering_tasks
                    SET {', '.join(update_fields)}
                    WHERE task_id = %s
                """, tuple(params))
                
                conn.commit()
                cursor.close()
                return_db_connection(conn)
                pg_success = True
        except Exception as e:
            logger.debug(f"更新PostgreSQL失败: {e}")
        
        # 如果PostgreSQL更新失败，更新文件系统
        if not pg_success:
            filepath = os.path.join(FEATURE_TASKS_DIR, f"{task_id}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                task["status"] = status
                task["updated_at"] = time.time()
                
                if progress is not None:
                    task["progress"] = progress
                
                if result is not None:
                    task["result"] = result
                
                if error_message is not None:
                    task["error_message"] = error_message
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(task, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"更新任务状态失败: {e}")
        return False
