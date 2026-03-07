#!/usr/bin/env python3
"""
ML模型管理器基础测试用例

测试ModelManager类的基本功能
"""

import pytest

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.legacy,
    pytest.mark.timeout(45),  # 45秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]
pytest.skip("legacy 模型管理器基础测试默认跳过，需手动启用", allow_module_level=True)
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from src.ml.model_manager import (
    ModelManager,
    ModelType,
    ModelStatus
)




class TestModelManagerBasic:
    """ML模型管理器基础测试类"""

    @pytest.fixture
    def temp_model_dir(self):
        """临时模型目录"""
        temp_dir = tempfile.mkdtemp(prefix="test_models_")
        yield temp_dir

        # 清理
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def model_manager(self, temp_model_dir):
        """模型管理器实例"""
        config = {
            'model_storage_path': temp_model_dir,
            'max_models_per_type': 5,
            'cache_max_size': 100
        }
        manager = ModelManager(config=config)
        return manager

    def test_initialization(self, model_manager, temp_model_dir):
        """测试初始化"""
        assert model_manager.config is not None
        assert model_manager.model_storage_path == temp_model_dir
        assert model_manager.max_models_per_type == 5
        assert isinstance(model_manager.models, dict)
        assert isinstance(model_manager.model_metadata, dict)
        assert isinstance(model_manager.active_models, dict)

    def test_model_type_enum(self):
        """测试模型类型枚举"""
        # 测试一些常见的模型类型
        assert ModelType.LINEAR_REGRESSION.value == "linear_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.LIGHTGBM.value == "lightgbm"

    def test_create_model(self, model_manager):
        """测试创建模型"""
        model_id = model_manager.create_model(
            model_name="test_linear_model",
            model_type=ModelType.LINEAR_REGRESSION,
            description="Test linear regression model"
        )

        assert model_id is not None
        assert model_id in model_manager.model_metadata

        metadata = model_manager.model_metadata[model_id]
        assert metadata.model_type == ModelType.LINEAR_REGRESSION
        assert metadata.model_name == "test_linear_model"
        assert metadata.status == ModelStatus.TRAINING

    def test_get_model(self, model_manager):
        """测试获取模型"""
        # 创建模型
        model_id = model_manager.create_model(
            model_name="test_rf_model",
            model_type=ModelType.RANDOM_FOREST
        )

        # 获取模型
        metadata = model_manager.model_metadata[model_id]

        assert metadata is not None
        assert metadata.model_id == model_id
        assert metadata.model_name == "test_rf_model"

    def test_get_nonexistent_model(self, model_manager):
        """测试获取不存在的模型"""
        metadata = model_manager.model_metadata.get("nonexistent_id")
        assert metadata is None

    def test_list_models(self, model_manager):
        """测试列出模型"""
        # 创建多个模型
        model1_id = model_manager.create_model(model_name="model1", model_type=ModelType.LINEAR_REGRESSION)
        model2_id = model_manager.create_model(model_name="model2", model_type=ModelType.RANDOM_FOREST)
        model3_id = model_manager.create_model(model_name="model3", model_type=ModelType.XGBOOST)

        models = model_manager.list_models()

        assert len(models) >= 3
        model_names = [m.name for m in models]
        assert "model1" in model_names
        assert "model2" in model_names
        assert "model3" in model_names

    def test_list_models_by_type(self, model_manager):
        """测试按类型列出模型"""
        # 创建不同类型的模型
        model_manager.create_model(model_name="lr1", model_type=ModelType.LINEAR_REGRESSION)
        model_manager.create_model(model_name="lr2", model_type=ModelType.LINEAR_REGRESSION)
        model_manager.create_model(model_name="rf1", model_type=ModelType.RANDOM_FOREST)

        lr_models = model_manager.list_models(model_type=ModelType.LINEAR_REGRESSION)
        rf_models = model_manager.list_models(model_type=ModelType.RANDOM_FOREST)

        assert len(lr_models) == 2
        assert len(rf_models) == 1

        lr_names = [m.name for m in lr_models]
        assert "lr1" in lr_names
        assert "lr2" in lr_names

    def test_update_model_status(self, model_manager):
        """测试更新模型状态"""
        # 创建模型
        model_id = model_manager.create_model(model_name="test_model", model_type=ModelType.LINEAR_REGRESSION)

        # 更新状态
        result = model_manager.update_model_status(model_id, ModelStatus.TRAINING)

        assert result is True
        metadata = model_manager.model_metadata.get(model_id)
        assert metadata.status == ModelStatus.TRAINING

    def test_update_nonexistent_model_status(self, model_manager):
        """测试更新不存在模型的状态"""
        result = model_manager.update_model_status("nonexistent_id", ModelStatus.TRAINED)
        assert result is False

    def test_delete_model(self, model_manager):
        """测试删除模型"""
        # 创建模型
        model_id = model_manager.create_model(model_name="test_model", model_type=ModelType.LINEAR_REGRESSION)

        # 验证模型存在
        assert model_manager.model_metadata.get(model_id) is not None

        # 删除模型
        result = model_manager.delete_model(model_id)

        assert result is True
        assert model_manager.model_metadata.get(model_id) is None

    def test_delete_nonexistent_model(self, model_manager):
        """测试删除不存在的模型"""
        result = model_manager.delete_model("nonexistent_id")
        assert result is False

    def test_model_storage_initialization(self, model_manager, temp_model_dir):
        """测试模型存储初始化"""
        # 验证存储目录被创建
        assert os.path.exists(temp_model_dir)

        metadata_dir = os.path.join(temp_model_dir, "metadata")
        assert os.path.exists(metadata_dir)

    def test_save_and_load_model_metadata(self, model_manager):
        """测试保存和加载模型元数据"""
        # 创建模型
        model_id = model_manager.create_model(
            model_name="test_model",
            model_type=ModelType.RANDOM_FOREST,
            description="Test model for save/load"
        )

        # 手动保存（通常是自动的）
        model_manager._save_model_metadata(model_id)

        # 验证元数据文件存在
        metadata_file = os.path.join(
            model_manager.metadata_storage_path,
            f"{model_id}.json"
        )
        assert os.path.exists(metadata_file)

        # 重新加载元数据
        model_manager._load_model_metadata(model_id)

        # 验证模型仍然存在
        metadata = model_manager.model_metadata.get(model_id)
        assert metadata is not None
        assert metadata.name == "test_model"

    def test_active_model_management(self, model_manager):
        """测试活跃模型管理"""
        # 创建两个相同类型的模型
        model1_id = model_manager.create_model(model_name="model1", model_type=ModelType.LINEAR_REGRESSION)
        model2_id = model_manager.create_model(model_name="model2", model_type=ModelType.LINEAR_REGRESSION)

        # 设置第一个为活跃模型
        result1 = model_manager.set_active_model(ModelType.LINEAR_REGRESSION, model1_id)
        assert result1 is True

        # 验证活跃模型
        active_id = model_manager.get_active_model(ModelType.LINEAR_REGRESSION)
        assert active_id == model1_id

        # 切换活跃模型
        result2 = model_manager.set_active_model(ModelType.LINEAR_REGRESSION, model2_id)
        assert result2 is True

        active_id2 = model_manager.get_active_model(ModelType.LINEAR_REGRESSION)
        assert active_id2 == model2_id

    def test_set_active_nonexistent_model(self, model_manager):
        """测试设置不存在的模型为活跃模型"""
        result = model_manager.set_active_model(ModelType.LINEAR_REGRESSION, "nonexistent_id")
        assert result is False

    def test_get_active_nonexistent_type(self, model_manager):
        """测试获取不存在类型模型的活跃模型"""
        active_id = model_manager.get_active_model(ModelType.XGBOOST)
        assert active_id is None

    def test_model_cache_operations(self, model_manager):
        """测试模型缓存操作"""
        # 缓存操作
        cache_key = "test_cache_key"
        cache_value = {"model": "data", "version": "1.0"}

        model_manager.prediction_cache[cache_key] = cache_value

        # 验证缓存
        assert cache_key in model_manager.prediction_cache
        assert model_manager.prediction_cache[cache_key] == cache_value

        # 清除缓存
        del model_manager.prediction_cache[cache_key]
        assert cache_key not in model_manager.prediction_cache

    def test_thread_safety(self, model_manager):
        """测试线程安全性"""
        import threading
        import queue

        results = queue.Queue()
        errors = []

        def worker(worker_id):
            try:
                # 每个worker创建一些模型
                for i in range(3):
                    model_id = model_manager.create_model(
                        model_name=f"worker_{worker_id}_model_{i}",
                        model_type=ModelType.LINEAR_REGRESSION
                    )
                    results.put(model_id)

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        model_ids = []
        while not results.empty():
            model_ids.append(results.get())

        assert len(model_ids) == 9  # 3 workers * 3 models each
        assert len(errors) == 0
        assert len(set(model_ids)) == 9  # 所有模型ID都唯一

        # 验证所有模型都存在
        all_models = model_manager.list_models()
        assert len(all_models) >= 9

    @pytest.mark.parametrize("model_type", [
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
    ])
    def test_model_creation_parametrized(self, model_manager, model_type):
        """参数化测试模型创建"""
        try:
            model_id = model_manager.create_model(
                model_name=f"test_{model_type.value}",
                model_type=model_type,
                description=f"Test {model_type.value} model"
            )

            assert model_id is not None
            metadata = model_manager.model_metadata.get(model_id)
            assert metadata.model_type == model_type
            assert metadata.model_name == f"test_{model_type.value}"
        except ValueError as e:
            # 对于某些模型类型，如果对应的库未安装，跳过测试
            if "库未安装" in str(e):
                pytest.skip(f"跳过测试：{e}")
            else:
                raise

    def test_model_metadata_persistence(self, model_manager):
        """测试模型元数据持久化"""
        # 创建模型并设置一些属性
        model_id = model_manager.create_model(
            ModelType.RANDOM_FOREST,
            "persistent_test_model",
            "Test model for persistence"
        )

        # 更新模型状态
        model_manager.update_model_status(model_id, ModelStatus.TRAINING)

        # 强制保存
        model_manager._save_model_metadata(model_id)

        # 清除内存中的元数据
        if model_id in model_manager.model_metadata:
            del model_manager.model_metadata[model_id]

        # 重新加载
        model_manager._load_model_metadata(model_id)

        # 验证持久化
        metadata = model_manager.model_metadata.get(model_id)
        assert metadata is not None
        assert metadata.name == "persistent_test_model"
        assert metadata.status == ModelStatus.TRAINING

    def test_max_models_per_type_limit(self, model_manager):
        """测试每种类型最大模型数量限制"""
        model_manager.max_models_per_type = 3

        # 创建超过限制数量的模型
        model_ids = []
        for i in range(5):  # 超过限制的3个
            model_id = model_manager.create_model(
                model_name=f"model_{i}",
                model_type=ModelType.LINEAR_REGRESSION
            )
            model_ids.append(model_id)

        # 验证只保留了最新的模型
        lr_models = model_manager.list_models(ModelType.LINEAR_REGRESSION)
        assert len(lr_models) <= model_manager.max_models_per_type

    def test_model_cleanup_operations(self, model_manager):
        """测试模型清理操作"""
        # 创建一些模型
        model_ids = []
        for i in range(3):
            model_id = model_manager.create_model(
                ModelType.LINEAR_REGRESSION,
                f"cleanup_model_{i}"
            )
            model_ids.append(model_id)

        initial_count = len(model_manager.list_models())

        # 删除一个模型
        model_manager.delete_model(model_ids[0])

        # 验证删除成功
        final_count = len(model_manager.list_models())
        assert final_count == initial_count - 1

        # 验证已删除的模型不存在
        assert model_manager.model_metadata.get(model_ids[0]) is None

        # 验证其他模型仍然存在
        for model_id in model_ids[1:]:
            assert model_manager.model_metadata.get(model_id) is not None
