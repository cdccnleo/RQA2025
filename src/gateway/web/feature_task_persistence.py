

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
