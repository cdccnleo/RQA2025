"""
模型持久化服务
负责模型文件的保存、加载和元数据管理
符合架构设计：数据管理层（模型持久化）
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
        logger.info(f"模型存储目录: {self.models_dir}")

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

            logger.info(f"模型文件已保存: {model_path}")

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

            logger.info(f"模型元数据已保存: {metadata_path}")

            # 保存到数据库
            success = self._save_to_database(
                model_id, job_id, model_type, str(model_path),
                metrics, hyperparameters, feature_columns, training_config
            )

            if success:
                logger.info(f"模型保存成功: {model_id}")
                return model_id
            else:
                logger.error(f"保存模型元数据到数据库失败: {model_id}")
                return None

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
                       training_time, epochs, trained_at, status, is_deployed,
                       model_path, hyperparameters, feature_columns, feature_count,
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
                    'is_deployed': row[15],
                    'model_path': row[16],
                    'hyperparameters': row[17],
                    'feature_columns': row[18],
                    'feature_count': row[19],
                    'training_data_source': row[20],
                    'training_data_range': row[21],
                    'training_samples': row[22],
                    'metadata': row[23],
                    'description': row[24],
                    'tags': row[25],
                    'version_notes': row[26]
                }
            return None

        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            return None

    def list_available_models(
        self,
        model_type: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        status: Optional[str] = 'active',
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        列出可用模型
        
        优先从PostgreSQL数据库获取，失败或为空时降级到文件系统

        Args:
            model_type: 模型类型筛选
            min_accuracy: 最小准确率筛选
            status: 状态筛选，None表示所有状态
            limit: 返回数量限制

        Returns:
            models: 模型列表
        """
        models = []
        
        # 第一步：尝试从数据库获取
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection

            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()

                # 构建查询
                query = """
                    SELECT model_id, model_name, model_type, model_version,
                           accuracy, loss, trained_at, status, is_deployed,
                           hyperparameters, feature_count, description
                    FROM trained_models
                """
                params = []
                
                # 添加状态筛选条件
                if status is not None:
                    query += " WHERE status = %s"
                    params.append(status)

                # 添加模型类型筛选
                if model_type:
                    if status is not None:
                        query += " AND model_type = %s"
                    else:
                        query += " WHERE model_type = %s"
                    params.append(model_type)

                # 添加最小准确率筛选
                if min_accuracy:
                    if status is not None or model_type:
                        query += " AND accuracy >= %s"
                    else:
                        query += " WHERE accuracy >= %s"
                    params.append(min_accuracy)

                query += " ORDER BY trained_at DESC LIMIT %s"
                params.append(limit)

                cursor.execute(query, params)

                rows = cursor.fetchall()
                cursor.close()
                return_db_connection(conn)

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
                        'is_deployed': row[8],
                        'hyperparameters': row[9],
                        'feature_count': row[10],
                        'description': row[11],
                        'source': 'database'
                    })

                if models:
                    logger.info(f"从数据库找到 {len(models)} 个可用模型")
                    return models
                else:
                    logger.warning("数据库中没有找到模型，将尝试从文件系统获取")
            else:
                logger.warning("数据库连接失败，将尝试从文件系统获取")
        except Exception as e:
            logger.warning(f"从数据库获取模型列表失败: {e}，将尝试从文件系统获取")
        
        # 第二步：数据库失败或为空，降级到文件系统
        try:
            if self.models_dir.exists():
                for model_dir in self.models_dir.iterdir():
                    if model_dir.is_dir():
                        model_file = model_dir / 'model.pkl'
                        metadata_file = model_dir / 'metadata.json'
                        
                        if model_file.exists():
                            # 读取元数据
                            model_info = {
                                'model_id': model_dir.name,
                                'model_name': model_dir.name,
                                'model_type': 'unknown',
                                'model_version': '1.0',
                                'accuracy': None,
                                'loss': None,
                                'trained_at': model_file.stat().st_ctime,
                                'status': 'active',
                                'is_deployed': False,
                                'hyperparameters': {},
                                'feature_count': 0,
                                'description': f'从文件系统加载的模型 {model_dir.name}',
                                'source': 'filesystem'
                            }
                            
                            # 尝试读取metadata.json
                            try:
                                if metadata_file.exists():
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                        model_info['model_type'] = metadata.get('model_type', 'unknown')
                                        model_info['model_name'] = metadata.get('model_id', model_dir.name)
                                        metrics = metadata.get('metrics', {})
                                        model_info['accuracy'] = metrics.get('accuracy')
                                        model_info['loss'] = metrics.get('loss')
                                        model_info['hyperparameters'] = metadata.get('hyperparameters', {})
                                        model_info['feature_count'] = len(metadata.get('feature_columns', []))
                            except Exception as e:
                                logger.debug(f"读取模型元数据失败 {model_dir.name}: {e}")
                            
                            # 应用筛选条件
                            if model_type and model_info['model_type'] != model_type:
                                continue
                            if min_accuracy and model_info['accuracy'] and model_info['accuracy'] < min_accuracy:
                                continue
                            
                            models.append(model_info)
                            
                            if len(models) >= limit:
                                break
                
                # 按训练时间排序
                models.sort(key=lambda x: x['trained_at'] if x['trained_at'] else 0, reverse=True)
                
                if models:
                    logger.info(f"从文件系统找到 {len(models)} 个可用模型")
        except Exception as e:
            logger.error(f"从文件系统获取模型也失败: {e}")
        
        return models

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

            logger.info(f"模型元数据已保存到数据库: {model_id}")
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
