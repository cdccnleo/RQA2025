#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强策略存储组件演示脚本
"""

from src.trading.strategy_workspace.store import StrategyStore
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStrategyStore(StrategyStore):
    """增强策略存储组件"""

    def __init__(self, storage_path: str = "data/strategies"):
        super().__init__(storage_path)

        # 创建增强目录
        (self.storage_path / "lineage").mkdir(exist_ok=True)
        (self.storage_path / "performance_history").mkdir(exist_ok=True)
        (self.storage_path / "templates").mkdir(exist_ok=True)

        # 加载血缘关系数据
        self.lineage_file = self.storage_path / "lineage" / "strategy_lineage.json"
        self.lineage_data = self._load_lineage_data()

    def _load_lineage_data(self) -> dict:
        """加载策略血缘关系数据"""
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载血缘关系数据失败: {e}")
        return {}

    def create_strategy_template(self, name: str, description: str,
                                 strategy_config: dict, parameters: dict,
                                 author: str, tags: list = None) -> str:
        """创建策略模板"""
        try:
            template_id = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            template_data = {
                "template_id": template_id,
                "name": name,
                "description": description,
                "strategy_config": strategy_config,
                "parameters": parameters,
                "author": author,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }

            template_file = self.storage_path / "templates" / f"{template_id}.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"创建策略模板成功: {template_id}")
            return template_id

        except Exception as e:
            logger.error(f"创建策略模板失败: {e}")
            raise

    def list_templates(self) -> list:
        """列出所有策略模板"""
        try:
            templates = []
            template_dir = self.storage_path / "templates"

            for template_file in template_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                templates.append(template_data)

            return templates

        except Exception as e:
            logger.error(f"列出策略模板失败: {e}")
            return []


def demo_enhanced_strategy_store():
    """演示增强策略存储组件功能"""
    print("=" * 60)
    print("增强策略存储组件演示")
    print("=" * 60)

    store = EnhancedStrategyStore()

    try:
        # 1. 创建策略模板
        print("\n1. 创建策略模板")
        print("-" * 30)

        template_config = {
            "nodes": [
                {"type": "data_source", "params": {"symbol": "000001.SZ"}},
                {"type": "feature", "params": {"indicators": ["MA", "RSI"]}},
                {"type": "model", "params": {"lookback": 20}},
                {"type": "trade", "params": {"threshold": 0.5}}
            ]
        }

        template_params = {
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 100},
            "threshold": {"type": "float", "default": 0.5, "min": 0.1, "max": 0.9}
        }

        template_id = store.create_strategy_template(
            name="基础趋势策略模板",
            description="适用于趋势市场的通用策略模板",
            strategy_config=template_config,
            parameters=template_params,
            author="系统",
            tags=["趋势", "通用", "模板"]
        )
        print(f"✅ 创建策略模板成功: {template_id}")

        # 2. 列出模板
        print("\n2. 列出策略模板")
        print("-" * 30)

        templates = store.list_templates()
        print(f"✅ 模板数量: {len(templates)}")
        for template in templates:
            print(f"  - {template['name']}: {template['description']}")

        print("\n" + "=" * 60)
        print("✅ 增强策略存储组件演示完成")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        logger.error(f"演示失败: {e}")


if __name__ == "__main__":
    demo_enhanced_strategy_store()
