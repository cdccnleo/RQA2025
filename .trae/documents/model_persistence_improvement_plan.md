# 模型持久化改进方案（方案一）

## 概述

本方案旨在建立完整的模型持久化机制，支持模型训练完成后的保存、版本管理、加载和查询，以满足策略回测和实盘交易对训练好模型的需求。

## 目标

1. 模型文件持久化到磁盘
2. 模型元数据存储到数据库
3. 支持模型版本管理
4. 支持模型查询和加载
5. 支持策略回测时的模型选择

## 实施计划

### 第一阶段：数据库设计（1小时）

#### 1.1 创建模型元数据表

**表名**: `trained_models`

```sql
-- 创建模型元数据表
CREATE TABLE IF NOT EXISTS trained_models (
    model_id VARCHAR(255) PRIMARY KEY,
    job_id VARCHAR(255) REFERENCES model_training_jobs(job_id),
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    model_path VARCHAR(500) NOT NULL,
    model_format VARCHAR(50) DEFAULT 'pickle',
    
    -- 性能指标
    accuracy FLOAT,
    loss FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    
    -- 训练信息
    training_time INTEGER,  -- 训练耗时（秒）
    epochs INTEGER,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 状态和配置
    status VARCHAR(50) DEFAULT 'active',  -- active, archived, deprecated
    is_deployed BOOLEAN DEFAULT FALSE,
    
    -- 超参数（JSON格式）
    hyperparameters JSONB,
    
    -- 特征信息
    feature_columns JSONB,  -- 特征列名列表
    feature_count INTEGER,
    target_column VARCHAR(255),
    
    -- 数据信息
    training_data_source VARCHAR(255),
    training_data_range JSONB,  -- {start_date, end_date}
    training_samples INTEGER,
    
    -- 元数据
    metadata JSONB,  -- 其他元数据
    description TEXT,
    tags JSONB,  -- 标签列表
    
    -- 版本控制
    parent_model_id VARCHAR(255),  -- 父模型ID（用于版本追踪）
    version_notes TEXT,
    
    -- 审计字段
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    -- 索引
    CONSTRAINT fk_job FOREIGN KEY (job_id) REFERENCES model_training_jobs(job_id) ON DELETE SET NULL
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_trained_models_job_id ON trained_models(job_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_model_type ON trained_models(model_type);
CREATE INDEX IF NOT EXISTS idx_trained_models_status ON trained_models(status);
CREATE INDEX IF NOT EXISTS idx_trained_models_trained_at ON trained_models(trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_accuracy ON trained_models(accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_is_deployed ON trained_models(is_deployed);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_trained_models_updated_at ON trained_models;
CREATE TRIGGER update_trained_models_updated_at
    BEFORE UPDATE ON trained_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

#### 1.2 修改现有表

**修改 `model_training_jobs` 表**:
```sql
-- 添加模型路径字段
ALTER TABLE model_training_jobs 
ADD COLUMN IF NOT EXISTS model_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS model_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS is_model_saved BOOLEAN DEFAULT FALSE;

-- 创建外键约束
ALTER TABLE model_training_jobs 
ADD CONSTRAINT fk_model_id 
FOREIGN KEY (model_id) REFERENCES trained_models(model_id) ON DELETE SET NULL;
```

### 第二阶段：模型保存服务（2小时）

#### 2.1 创建模型持久化服务

**文件**: `src/gateway/web/model_persistence_service.py`

```python
"""
模型持久化服务
负责模型文件的保存、加载和元数据管理
"""

import os
import pickle
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 模型存储路径
MODELS_DIR = os.getenv('MODELS_DIR', '/app/models')

class ModelPersistenceService:
    """模型持久化服务"""
    
    def __init__(self):
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(
        self,
        model: Any,
        job_id: str,
        model_type: str,
        metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        feature_columns: List[str],
        training_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        保存模型到磁盘和数据库
        
        Args:
            model: 训练好的模型对象
            job_id: 训练任务ID
            model_type: 模型类型
            metrics: 训练指标
            hyperparameters: 超参数
            feature_columns: 特征列名
            training_config: 训练配置
            
        Returns:
            model_id: 模型ID，保存失败返回None
        """
        try:
            # 生成模型ID
            model_id = f"model_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 创建模型目录
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # 保存模型文件
            model_path = model_dir / 'model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存模型元数据
            metadata = {
                'model_id': model_id,
                'job_id': job_id,
                'model_type': model_type,
                'metrics': metrics,
                'hyperparameters': hyperparameters,
                'feature_columns': feature_columns,
                'training_config': training_config,
                'saved_at': datetime.now().isoformat()
            }
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 保存到数据库
            self._save_to_database(model_id, job_id, model_type, str(model_path), metrics, hyperparameters, feature_columns, training_config)
            
            logger.info(f"模型保存成功: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return None
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        加载模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            model: 模型对象，加载失败返回None
        """
        try:
            # 从数据库获取模型路径
            model_path = self._get_model_path_from_db(model_id)
            if not model_path:
                logger.error(f"模型未找到: {model_id}")
                return None
            
            # 加载模型文件
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"模型加载成功: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型元数据"""
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return None
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, job_id, model_name, model_type, model_version,
                       accuracy, loss, precision, recall, f1_score, auc_roc,
                       training_time, epochs, trained_at, status,
                       hyperparameters, feature_columns, feature_count,
                       training_data_source, training_data_range, training_samples,
                       metadata, description, tags, version_notes
                FROM trained_models
                WHERE model_id = %s
            """, (model_id,))
            
            row = cursor.fetchone()
            cursor.close()
            return_db_connection(conn)
            
            if row:
                return {
                    'model_id': row[0],
                    'job_id': row[1],
                    'model_name': row[2],
                    'model_type': row[3],
                    'model_version': row[4],
                    'accuracy': row[5],
                    'loss': row[6],
                    'precision': row[7],
                    'recall': row[8],
                    'f1_score': row[9],
                    'auc_roc': row[10],
                    'training_time': row[11],
                    'epochs': row[12],
                    'trained_at': row[13],
                    'status': row[14],
                    'hyperparameters': row[15],
                    'feature_columns': row[16],
                    'feature_count': row[17],
                    'training_data_source': row[18],
                    'training_data_range': row[19],
                    'training_samples': row[20],
                    'metadata': row[21],
                    'description': row[22],
                    'tags': row[23],
                    'version_notes': row[24]
                }
            return None
            
        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            return None
    
    def list_available_models(
        self,
        model_type: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        status: str = 'active',
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        列出可用模型
        
        Args:
            model_type: 模型类型筛选
            min_accuracy: 最小准确率筛选
            status: 状态筛选
            limit: 返回数量限制
            
        Returns:
            models: 模型列表
        """
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            # 构建查询
            query = """
                SELECT model_id, model_name, model_type, model_version,
                       accuracy, loss, trained_at, status,
                       hyperparameters, feature_count, description
                FROM trained_models
                WHERE status = %s
            """
            params = [status]
            
            if model_type:
                query += " AND model_type = %s"
                params.append(model_type)
            
            if min_accuracy:
                query += " AND accuracy >= %s"
                params.append(min_accuracy)
            
            query += " ORDER BY trained_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            cursor.close()
            return_db_connection(conn)
            
            models = []
            for row in rows:
                models.append({
                    'model_id': row[0],
                    'model_name': row[1],
                    'model_type': row[2],
                    'model_version': row[3],
                    'accuracy': row[4],
                    'loss': row[5],
                    'trained_at': row[6],
                    'status': row[7],
                    'hyperparameters': row[8],
                    'feature_count': row[9],
                    'description': row[10]
                })
            
            return models
            
        except Exception as e:
            logger.error(f"列出可用模型失败: {e}")
            return []
    
    def _save_to_database(
        self,
        model_id: str,
        job_id: str,
        model_type: str,
        model_path: str,
        metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        feature_columns: List[str],
        training_config: Dict[str, Any]
    ) -> bool:
        """保存模型元数据到数据库"""
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 准备数据
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            accuracy = metrics.get('accuracy')
            loss = metrics.get('loss')
            precision = metrics.get('precision')
            recall = metrics.get('recall')
            f1_score = metrics.get('f1_score')
            auc_roc = metrics.get('auc_roc')
            training_time = metrics.get('training_time')
            epochs = hyperparameters.get('epochs', training_config.get('epochs', 100))
            feature_count = len(feature_columns) if feature_columns else 0
            
            # 插入数据
            cursor.execute("""
                INSERT INTO trained_models (
                    model_id, job_id, model_name, model_type, model_path,
                    accuracy, loss, precision, recall, f1_score, auc_roc,
                    training_time, epochs, hyperparameters, feature_columns, feature_count,
                    training_data_source, training_data_range, training_samples,
                    metadata, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_id, job_id, model_name, model_type, model_path,
                accuracy, loss, precision, recall, f1_score, auc_roc,
                training_time, epochs,
                json.dumps(hyperparameters),
                json.dumps(feature_columns),
                feature_count,
                training_config.get('data_source'),
                json.dumps(training_config.get('time_range')),
                training_config.get('training_samples'),
                json.dumps({'training_config': training_config}),
                f"{model_type} model trained on {datetime.now().strftime('%Y-%m-%d')}"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            return True
            
        except Exception as e:
            logger.error(f"保存模型元数据到数据库失败: {e}")
            return False
    
    def _get_model_path_from_db(self, model_id: str) -> Optional[str]:
        """从数据库获取模型路径"""
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return None
            
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model_path FROM trained_models WHERE model_id = %s",
                (model_id,)
            )
            
            row = cursor.fetchone()
            cursor.close()
            return_db_connection(conn)
            
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"获取模型路径失败: {e}")
            return None


# 单例实例
_model_persistence_service = None

def get_model_persistence_service() -> ModelPersistenceService:
    """获取模型持久化服务实例"""
    global _model_persistence_service
    if _model_persistence_service is None:
        _model_persistence_service = ModelPersistenceService()
    return _model_persistence_service
```

### 第三阶段：修改训练任务执行器（1小时）

#### 3.1 修改 `training_job_executor.py`

在 `_process_training_job` 方法中添加模型保存逻辑：

```python
async def _process_training_job(self, task):
    """处理训练任务"""
    # ... 现有代码 ...
    
    try:
        # ... 训练逻辑 ...
        
        # 训练完成后保存模型
        if result.get('accuracy') and result['accuracy'] > 0.5:  # 准确率大于50%才保存
            try:
                from .model_persistence_service import get_model_persistence_service
                
                persistence_service = get_model_persistence_service()
                
                # 准备模型对象（这里需要从实际训练器获取）
                model_object = self._get_trained_model()
                
                if model_object:
                    model_id = persistence_service.save_model(
                        model=model_object,
                        job_id=original_job_id,
                        model_type=model_type,
                        metrics=result,
                        hyperparameters=config,
                        feature_columns=config.get('feature_columns', []),
                        training_config=config
                    )
                    
                    if model_id:
                        logger.info(f"模型已保存: {model_id}")
                        result['model_id'] = model_id
                        
                        # 更新训练任务，关联模型ID
                        await self._update_job_status(
                            original_job_id, "completed",
                            model_id=model_id,
                            is_model_saved=True
                        )
                    else:
                        logger.warning("模型保存失败")
                else:
                    logger.warning("没有可保存的模型对象")
                    
            except Exception as e:
                logger.error(f"保存模型时出错: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"处理训练任务失败: {e}")
        raise

def _get_trained_model(self):
    """获取训练好的模型对象"""
    # 这里需要从实际的模型训练器获取训练好的模型
    # 暂时返回None，需要根据实际情况实现
    if self.model_trainer and hasattr(self.model_trainer, 'get_model'):
        return self.model_trainer.get_model()
    return None
```

### 第四阶段：修改回测服务（1小时）

#### 4.1 修改 `backtest_service.py`

修改 `get_available_models_for_backtest` 函数：

```python
async def get_available_models_for_backtest() -> List[Dict[str, Any]]:
    """
    获取可用于回测的模型列表
    
    Returns:
        可用模型列表
    """
    try:
        # 使用模型持久化服务获取可用模型
        from .model_persistence_service import get_model_persistence_service
        
        persistence_service = get_model_persistence_service()
        models = persistence_service.list_available_models(
            status='active',
            min_accuracy=0.5,  # 只返回准确率大于50%的模型
            limit=100
        )
        
        if models:
            logger.info(f"找到 {len(models)} 个可用于回测的模型")
            
            # 格式化模型信息
            available_models = []
            for model in models:
                available_models.append({
                    "model_id": model['model_id'],
                    "model_name": model['model_name'],
                    "model_type": model['model_type'],
                    "accuracy": model['accuracy'],
                    "loss": model['loss'],
                    "trained_at": model['trained_at'].isoformat() if hasattr(model['trained_at'], 'isoformat') else model['trained_at'],
                    "feature_count": model['feature_count'],
                    "hyperparameters": model['hyperparameters'],
                    "description": model['description'] or f"{model['model_type']} - 准确率: {model['accuracy']:.2%}"
                })
            
            return available_models
        
        # 如果没有找到模型，返回空列表
        logger.info("没有找到可用于回测的模型")
        return []
        
    except Exception as e:
        logger.error(f"获取可用模型列表失败: {e}")
        return []
```

### 第五阶段：创建模型管理API（1小时）

#### 5.1 创建模型管理路由

**文件**: `src/gateway/web/model_management_routes.py`

```python
"""
模型管理API路由
提供模型查询、加载、删除等接口
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/models", tags=["models"])
async def list_models(
    model_type: Optional[str] = None,
    status: str = 'active',
    min_accuracy: Optional[float] = None,
    limit: int = 100
):
    """列出所有模型"""
    try:
        from .model_persistence_service import get_model_persistence_service
        
        service = get_model_persistence_service()
        models = service.list_available_models(
            model_type=model_type,
            status=status,
            min_accuracy=min_accuracy,
            limit=limit
        )
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出模型失败: {str(e)}")

@router.get("/models/{model_id}", tags=["models"])
async def get_model_details(model_id: str):
    """获取模型详情"""
    try:
        from .model_persistence_service import get_model_persistence_service
        
        service = get_model_persistence_service()
        metadata = service.get_model_metadata(model_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="模型未找到")
        
        return {
            "success": True,
            "model": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")

@router.post("/models/{model_id}/load", tags=["models"])
async def load_model(model_id: str):
    """加载模型"""
    try:
        from .model_persistence_service import get_model_persistence_service
        
        service = get_model_persistence_service()
        model = service.load_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="模型加载失败")
        
        return {
            "success": True,
            "message": "模型加载成功",
            "model_id": model_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")

@router.delete("/models/{model_id}", tags=["models"])
async def delete_model(model_id: str):
    """删除模型"""
    try:
        from .model_persistence_service import get_model_persistence_service
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        # 从数据库删除
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE trained_models SET status = 'deleted' WHERE model_id = %s",
                (model_id,)
            )
            conn.commit()
            cursor.close()
            return_db_connection(conn)
        
        return {
            "success": True,
            "message": "模型已删除"
        }
        
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")
```

### 第六阶段：数据库迁移脚本（30分钟）

#### 6.1 创建迁移脚本

**文件**: `scripts/migrations/001_create_trained_models_table.sql`

```sql
-- 迁移脚本：创建模型元数据表
-- 执行时间: 2026-02-16

BEGIN;

-- 创建模型元数据表
CREATE TABLE IF NOT EXISTS trained_models (
    model_id VARCHAR(255) PRIMARY KEY,
    job_id VARCHAR(255),
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    model_path VARCHAR(500) NOT NULL,
    model_format VARCHAR(50) DEFAULT 'pickle',
    
    -- 性能指标
    accuracy FLOAT,
    loss FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    
    -- 训练信息
    training_time INTEGER,
    epochs INTEGER,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 状态和配置
    status VARCHAR(50) DEFAULT 'active',
    is_deployed BOOLEAN DEFAULT FALSE,
    
    -- JSON字段
    hyperparameters JSONB,
    feature_columns JSONB,
    training_data_source VARCHAR(255),
    training_data_range JSONB,
    training_samples INTEGER,
    metadata JSONB,
    description TEXT,
    tags JSONB,
    
    -- 版本控制
    parent_model_id VARCHAR(255),
    version_notes TEXT,
    
    -- 审计字段
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_trained_models_job_id ON trained_models(job_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_model_type ON trained_models(model_type);
CREATE INDEX IF NOT EXISTS idx_trained_models_status ON trained_models(status);
CREATE INDEX IF NOT EXISTS idx_trained_models_trained_at ON trained_models(trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_accuracy ON trained_models(accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_is_deployed ON trained_models(is_deployed);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_trained_models_updated_at ON trained_models;
CREATE TRIGGER update_trained_models_updated_at
    BEFORE UPDATE ON trained_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 修改 model_training_jobs 表
ALTER TABLE model_training_jobs 
ADD COLUMN IF NOT EXISTS model_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS model_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS is_model_saved BOOLEAN DEFAULT FALSE;

-- 添加外键约束（可选，如果不需要严格约束可以注释掉）
-- ALTER TABLE model_training_jobs 
-- ADD CONSTRAINT fk_model_id 
-- FOREIGN KEY (model_id) REFERENCES trained_models(model_id) ON DELETE SET NULL;

COMMIT;
```

## 实施优先级

| 阶段 | 任务 | 优先级 | 预计时间 | 依赖 |
|------|------|--------|----------|------|
| 第一阶段 | 数据库设计 | 🔴 高 | 1小时 | 无 |
| 第二阶段 | 模型持久化服务 | 🔴 高 | 2小时 | 第一阶段 |
| 第三阶段 | 修改训练任务执行器 | 🔴 高 | 1小时 | 第二阶段 |
| 第四阶段 | 修改回测服务 | 🔴 高 | 1小时 | 第二阶段 |
| 第五阶段 | 模型管理API | 🟡 中 | 1小时 | 第二阶段 |
| 第六阶段 | 数据库迁移 | 🔴 高 | 30分钟 | 第一阶段 |

## 验收标准

### 功能验收
- [ ] 训练完成的模型自动保存到磁盘
- [ ] 模型元数据存储到 `trained_models` 表
- [ ] 可以通过API查询可用模型列表
- [ ] 策略回测页面可以显示训练好的模型
- [ ] 可以选择模型进行回测
- [ ] 模型可以正确加载用于预测

### 性能验收
- [ ] 模型保存时间 < 5秒
- [ ] 模型加载时间 < 3秒
- [ ] 查询模型列表时间 < 1秒

### 数据完整性验收
- [ ] 模型文件和元数据一致
- [ ] 训练任务和模型正确关联
- [ ] 模型版本信息正确记录

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型文件过大 | 磁盘空间不足 | 设置模型文件大小限制，定期清理旧版本 |
| 模型格式不兼容 | 无法加载 | 记录模型格式版本，提供格式转换工具 |
| 数据库性能问题 | 查询慢 | 添加适当的索引，考虑分页查询 |
| 并发保存冲突 | 数据不一致 | 使用事务和锁机制 |
