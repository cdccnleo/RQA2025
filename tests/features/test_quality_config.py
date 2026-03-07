"""
测试自定义评分配置管理
"""

import unittest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.feature_quality_config import (
    FeatureQualityConfigManager,
    UserQualityConfig,
    get_config_manager,
    get_user_custom_score
)


class TestFeatureQualityConfigManager(unittest.TestCase):
    """测试特征质量配置管理器"""

    def setUp(self):
        """测试前准备"""
        self.manager = FeatureQualityConfigManager()
        self.manager.clear_all_cache()
        self.test_user_id = "test_user_123"

    def test_create_config(self):
        """测试创建配置"""
        config = self.manager.create_config(
            user_id=self.test_user_id,
            feature_name="SMA_5",
            custom_score=0.85,
            reason="测试原因"
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.user_id, self.test_user_id)
        self.assertEqual(config.feature_name, "SMA_5")
        self.assertEqual(config.custom_score, 0.85)
        self.assertEqual(config.reason, "测试原因")
        self.assertTrue(config.is_active)

    def test_create_config_invalid_score(self):
        """测试创建配置时评分无效"""
        # 评分超出范围
        config = self.manager.create_config(
            user_id=self.test_user_id,
            feature_name="SMA_5",
            custom_score=1.5  # 无效
        )
        self.assertIsNone(config)

        config = self.manager.create_config(
            user_id=self.test_user_id,
            feature_name="SMA_5",
            custom_score=-0.5  # 无效
        )
        self.assertIsNone(config)

    def test_get_user_configs(self):
        """测试获取用户配置列表"""
        # 创建多个配置
        self.manager.create_config(self.test_user_id, "SMA_5", 0.85)
        self.manager.create_config(self.test_user_id, "RSI", 0.80)
        self.manager.create_config(self.test_user_id, "BOLL_upper", 0.75)

        configs = self.manager.get_user_configs(self.test_user_id)

        self.assertEqual(len(configs), 3)
        feature_names = [c.feature_name for c in configs]
        self.assertIn("SMA_5", feature_names)
        self.assertIn("RSI", feature_names)
        self.assertIn("BOLL_upper", feature_names)

    def test_get_config_by_feature(self):
        """测试获取特定特征的配置"""
        self.manager.create_config(self.test_user_id, "SMA_5", 0.85, "测试")

        config = self.manager.get_config_by_feature(self.test_user_id, "SMA_5")

        self.assertIsNotNone(config)
        self.assertEqual(config.feature_name, "SMA_5")
        self.assertEqual(config.custom_score, 0.85)

    def test_update_config(self):
        """测试更新配置"""
        # 先创建配置
        config = self.manager.create_config(
            self.test_user_id, "SMA_5", 0.85, "原始原因"
        )

        # 更新配置
        updated = self.manager.update_config(
            config.config_id,
            custom_score=0.90,
            reason="更新原因"
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.custom_score, 0.90)
        self.assertEqual(updated.reason, "更新原因")

    def test_delete_config(self):
        """测试删除配置"""
        # 先创建配置
        config = self.manager.create_config(self.test_user_id, "SMA_5", 0.85)

        # 删除配置
        result = self.manager.delete_config(config.config_id)
        self.assertTrue(result)

        # 确认已删除
        configs = self.manager.get_user_configs(self.test_user_id)
        self.assertEqual(len(configs), 0)

    def test_reset_to_default(self):
        """测试重置为默认"""
        # 先创建配置
        self.manager.create_config(self.test_user_id, "SMA_5", 0.85)

        # 重置
        result = self.manager.reset_to_default(self.test_user_id, "SMA_5")
        self.assertTrue(result)

        # 确认已禁用
        config = self.manager.get_config_by_feature(self.test_user_id, "SMA_5")
        self.assertIsNone(config)  # 因为is_active=FALSE，所以查不到

    def test_batch_create_configs(self):
        """测试批量创建配置"""
        configs = [
            {"feature_name": "SMA_5", "custom_score": 0.85, "reason": "批量1"},
            {"feature_name": "RSI", "custom_score": 0.80, "reason": "批量2"},
            {"feature_name": "BOLL_upper", "custom_score": 0.75, "reason": "批量3"}
        ]

        result = self.manager.batch_create_configs(self.test_user_id, configs)

        self.assertEqual(len(result['success']), 3)
        self.assertEqual(len(result['failed']), 0)
        self.assertEqual(result['total'], 3)

    def test_cache_functionality(self):
        """测试缓存功能"""
        # 创建配置
        self.manager.create_config(self.test_user_id, "SMA_5", 0.85)

        # 第一次获取（应该缓存）
        configs1 = self.manager.get_user_configs(self.test_user_id)

        # 第二次获取（应该使用缓存）
        configs2 = self.manager.get_user_configs(self.test_user_id)

        self.assertEqual(len(configs1), len(configs2))

    def test_update_nonexistent_config(self):
        """测试更新不存在的配置"""
        result = self.manager.update_config(99999, custom_score=0.90)
        self.assertIsNone(result)

    def test_delete_nonexistent_config(self):
        """测试删除不存在的配置"""
        result = self.manager.delete_config(99999)
        self.assertFalse(result)


class TestGetUserCustomScore(unittest.TestCase):
    """测试获取用户自定义评分便捷函数"""

    def setUp(self):
        """测试前准备"""
        self.manager = get_config_manager()
        self.manager.clear_all_cache()
        self.test_user_id = "test_user_456"

    def test_get_custom_score_exists(self):
        """测试获取存在的自定义评分"""
        self.manager.create_config(self.test_user_id, "SMA_5", 0.85)

        score = get_user_custom_score(self.test_user_id, "SMA_5")

        self.assertIsNotNone(score)
        self.assertEqual(score, 0.85)

    def test_get_custom_score_not_exists(self):
        """测试获取不存在的自定义评分"""
        score = get_user_custom_score(self.test_user_id, "NONEXISTENT")

        self.assertIsNone(score)

    def test_get_custom_score_inactive(self):
        """测试获取已禁用的自定义评分"""
        # 创建配置
        config = self.manager.create_config(self.test_user_id, "SMA_5", 0.85)

        # 禁用配置
        self.manager.update_config(config.config_id, is_active=False)

        # 应该返回None，因为配置已禁用
        score = get_user_custom_score(self.test_user_id, "SMA_5")
        self.assertIsNone(score)


class TestQualityScorerWithCustom(unittest.TestCase):
    """测试支持自定义评分的评分器"""

    def setUp(self):
        """测试前准备"""
        from src.gateway.web.feature_quality_config import get_config_manager
        self.manager = get_config_manager()
        self.manager.clear_all_cache()
        self.test_user_id = "test_user_789"

    def test_get_feature_quality_score_with_custom(self):
        """测试获取评分（有自定义）"""
        from src.features.quality import get_feature_quality_score_with_custom

        # 创建自定义评分
        self.manager.create_config(self.test_user_id, "SMA_5", 0.75)

        # 获取评分（应该返回自定义评分）
        score = get_feature_quality_score_with_custom("SMA_5", self.test_user_id)

        self.assertEqual(score, 0.75)  # 自定义评分，不是默认的0.90

    def test_get_feature_quality_score_without_custom(self):
        """测试获取评分（无自定义）"""
        from src.features.quality import get_feature_quality_score_with_custom

        # 不创建自定义评分
        score = get_feature_quality_score_with_custom("SMA_5", self.test_user_id)

        self.assertEqual(score, 0.90)  # 默认评分

    def test_calculate_final_quality_score_with_custom(self):
        """测试计算最终评分（有自定义）"""
        from src.features.quality import calculate_final_quality_score_with_custom

        # 创建自定义评分
        self.manager.create_config(self.test_user_id, "SMA_5", 0.75)

        # 计算评分（应该返回自定义评分，忽略数据质量因子）
        score = calculate_final_quality_score_with_custom(
            "SMA_5", data_quality_factor=0.95, user_id=self.test_user_id
        )

        self.assertEqual(score, 0.75)  # 自定义评分

    def test_calculate_final_quality_score_without_custom(self):
        """测试计算最终评分（无自定义）"""
        from src.features.quality import calculate_final_quality_score_with_custom

        # 不创建自定义评分
        score = calculate_final_quality_score_with_custom(
            "SMA_5", data_quality_factor=0.95, user_id=self.test_user_id
        )

        self.assertEqual(score, 0.855)  # 0.90 * 0.95


if __name__ == '__main__':
    unittest.main()
