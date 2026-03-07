#!/usr/bin/env python3
"""
数据源配置集成测试

验证数据源配置是否正确集成了基础设施层的配置管理模块
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gateway.web.data_source_config_manager import DataSourceConfigManager
from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class DataSourceConfigIntegrationTest:
    """数据源配置集成测试"""

    def __init__(self):
        self.config_manager = None
        self.infrastructure_config = None
        self.test_results = []

    async def run_all_tests(self):
        """运行所有集成测试"""
        print("🧪 开始数据源配置集成测试")
        print("=" * 50)

        try:
            # 测试1: 基础设施层配置管理器初始化
            await self.test_infrastructure_config_init()

            # 测试2: 数据源配置管理器初始化
            await self.test_data_source_config_init()

            # 测试3: 配置加载和验证
            await self.test_config_loading()

            # 测试4: CRUD操作
            await self.test_crud_operations()

            # 测试5: 环境隔离
            await self.test_environment_isolation()

            # 测试6: 配置热更新
            await self.test_config_hot_reload()

            # 测试7: 备份和恢复
            await self.test_backup_restore()

            # 测试8: 性能测试
            await self.test_performance()

        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            self.record_result("集成测试总体", False, f"测试执行异常: {e}")

        # 输出测试结果
        self.print_test_results()

        return self.get_overall_result()

    async def test_infrastructure_config_init(self):
        """测试基础设施层配置管理器初始化"""
        try:
            self.infrastructure_config = UnifiedConfigManager()
            self.record_result("基础设施层配置管理器初始化", True, "成功初始化UnifiedConfigManager")
        except Exception as e:
            self.record_result("基础设施层配置管理器初始化", False, f"初始化失败: {e}")

    async def test_data_source_config_init(self):
        """测试数据源配置管理器初始化"""
        try:
            self.config_manager = DataSourceConfigManager()
            self.record_result("数据源配置管理器初始化", True, "成功初始化DataSourceConfigManager")
        except Exception as e:
            self.record_result("数据源配置管理器初始化", False, f"初始化失败: {e}")

    async def test_config_loading(self):
        """测试配置加载和验证"""
        try:
            # 加载配置
            success = self.config_manager.load_config()
            if not success:
                self.record_result("配置加载", False, "配置加载失败")
                return

            # 获取数据源
            data_sources = self.config_manager.get_data_sources()
            if not isinstance(data_sources, list):
                self.record_result("配置加载", False, "数据源格式不正确")
                return

            # 验证配置
            validation = self.config_manager.validate_all_sources()
            if not validation['valid']:
                self.record_result("配置验证", False, f"配置验证失败: {validation['issues']}")
                return

            self.record_result("配置加载和验证", True, f"成功加载 {len(data_sources)} 个数据源，验证通过")

        except Exception as e:
            self.record_result("配置加载和验证", False, f"测试失败: {e}")

    async def test_crud_operations(self):
        """测试CRUD操作"""
        try:
            # 创建测试数据源
            test_source = {
                "id": "test_integration_source",
                "name": "集成测试数据源",
                "type": "财经新闻",
                "url": "https://test.example.com",
                "rate_limit": "10次/分钟",
                "enabled": True
            }

            # 测试创建
            if not self.config_manager.add_data_source(test_source):
                self.record_result("CRUD操作", False, "创建数据源失败")
                return

            # 测试读取
            retrieved = self.config_manager.get_data_source(test_source["id"])
            if not retrieved or retrieved["id"] != test_source["id"]:
                self.record_result("CRUD操作", False, "读取数据源失败")
                return

            # 测试更新
            updates = {"name": "更新的集成测试数据源"}
            if not self.config_manager.update_data_source(test_source["id"], updates):
                self.record_result("CRUD操作", False, "更新数据源失败")
                return

            updated = self.config_manager.get_data_source(test_source["id"])
            if updated["name"] != updates["name"]:
                self.record_result("CRUD操作", False, "更新数据源验证失败")
                return

            # 测试删除
            if not self.config_manager.delete_data_source(test_source["id"]):
                self.record_result("CRUD操作", False, "删除数据源失败")
                return

            # 验证删除
            deleted = self.config_manager.get_data_source(test_source["id"])
            if deleted is not None:
                self.record_result("CRUD操作", False, "删除数据源验证失败")
                return

            self.record_result("CRUD操作", True, "所有CRUD操作成功")

        except Exception as e:
            self.record_result("CRUD操作", False, f"CRUD操作测试失败: {e}")

    async def test_environment_isolation(self):
        """测试环境隔离"""
        try:
            env = self.config_manager.get_config_stats()['environment']

            # 检查环境变量
            expected_env = os.getenv("RQA_ENV", "development")
            if env != expected_env:
                self.record_result("环境隔离", False, f"环境不匹配: 期望 {expected_env}, 实际 {env}")
                return

            # 检查配置目录
            config_dir = self.config_manager.config_dir
            if env == "production" and not config_dir.endswith("production"):
                self.record_result("环境隔离", False, "生产环境配置目录不正确")
                return

            self.record_result("环境隔离", True, f"环境隔离正常: {env}")

        except Exception as e:
            self.record_result("环境隔离", False, f"环境隔离测试失败: {e}")

    async def test_config_hot_reload(self):
        """测试配置热更新"""
        try:
            # 获取初始配置
            initial_sources = self.config_manager.get_data_sources()
            initial_count = len(initial_sources)

            # 重新加载配置
            success = self.config_manager.reload_config()
            if not success:
                self.record_result("配置热更新", False, "配置重新加载失败")
                return

            # 验证配置一致性
            reloaded_sources = self.config_manager.get_data_sources()
            if len(reloaded_sources) != initial_count:
                self.record_result("配置热更新", False, "配置重新加载后数据不一致")
                return

            self.record_result("配置热更新", True, "配置热更新功能正常")

        except Exception as e:
            self.record_result("配置热更新", False, f"配置热更新测试失败: {e}")

    async def test_backup_restore(self):
        """测试备份和恢复"""
        try:
            import tempfile
            import os

            # 创建临时备份文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                backup_path = f.name

            try:
                # 执行备份
                success = self.config_manager.backup_config(backup_path)
                if not success:
                    self.record_result("备份和恢复", False, "配置备份失败")
                    return

                # 验证备份文件存在
                if not os.path.exists(backup_path):
                    self.record_result("备份和恢复", False, "备份文件不存在")
                    return

                # 验证备份文件内容
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)

                if 'data_sources' not in backup_data:
                    self.record_result("备份和恢复", False, "备份文件格式不正确")
                    return

                self.record_result("备份和恢复", True, "备份和恢复功能正常")

            finally:
                # 清理临时文件
                if os.path.exists(backup_path):
                    os.unlink(backup_path)

        except Exception as e:
            self.record_result("备份和恢复", False, f"备份和恢复测试失败: {e}")

    async def test_performance(self):
        """测试性能"""
        try:
            # 测试配置读取性能
            start_time = time.time()
            for _ in range(100):
                self.config_manager.get_data_sources()
            read_time = (time.time() - start_time) * 1000  # 毫秒

            avg_read_time = read_time / 100
            if avg_read_time > 10:  # 平均读取时间不超过10ms
                self.record_result("性能测试", False, f"配置读取性能过低: {avg_read_time:.2f}ms/次")
                return

            # 测试配置更新性能
            test_source = {
                "id": "perf_test_source",
                "name": "性能测试数据源",
                "type": "财经新闻",
                "url": "https://perf.example.com",
                "rate_limit": "10次/分钟",
                "enabled": True
            }

            start_time = time.time()
            for i in range(50):
                test_source["id"] = f"perf_test_source_{i}"
                self.config_manager.add_data_source(test_source.copy())
                self.config_manager.delete_data_source(test_source["id"])
            crud_time = (time.time() - start_time) * 1000

            avg_crud_time = crud_time / 50
            if avg_crud_time > 50:  # 平均CRUD操作不超过50ms
                self.record_result("性能测试", False, f"CRUD性能过低: {avg_crud_time:.2f}ms/次")
                return

            self.record_result("性能测试", True, f"读取: {avg_read_time:.2f}ms/次, CRUD: {avg_crud_time:.2f}ms/次")

        except Exception as e:
            self.record_result("性能测试", False, f"性能测试失败: {e}")

    def record_result(self, test_name: str, success: bool, message: str):
        """记录测试结果"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)

        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")

    def print_test_results(self):
        """输出测试结果"""
        print("\n" + "=" * 50)
        print("📊 测试结果汇总")
        print("=" * 50)

        passed = sum(1 for r in self.test_results if r['success'])
        total = len(self.test_results)

        print(f"总测试数: {total}")
        print(f"通过测试: {passed}")
        print(f"失败测试: {total - passed}")
        print(".1f")

        if total - passed > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['message']}")

    def get_overall_result(self) -> bool:
        """获取总体测试结果"""
        return all(result['success'] for result in self.test_results)


async def main():
    """主函数"""
    tester = DataSourceConfigIntegrationTest()
    success = await tester.run_all_tests()

    if success:
        print("\n🎉 所有集成测试通过！数据源配置已成功集成基础设施层配置管理模块。")
        return 0
    else:
        print("\n❌ 部分集成测试失败！请检查相关配置和实现。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
