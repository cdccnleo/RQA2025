#!/usr/bin/env python3
"""
RQA2025 数据源配置迁移脚本
安全地将数据源配置迁移到环境特定的目录中
"""

import os
import json
import shutil
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSourceMigrator:
    """数据源配置迁移器"""

    def __init__(self):
        self.base_dir = Path("data")
        self.old_config = self.base_dir / "data_sources_config.json"
        self.backup_dir = self.base_dir / "backups"

    def detect_environment(self):
        """检测当前运行环境"""
        env = os.getenv("RQA_ENV", "development").lower()
        logger.info(f"检测到环境: {env}")
        return env

    def get_target_config_path(self, env):
        """根据环境获取目标配置文件路径"""
        if env == "production":
            return self.base_dir / "production" / "data_sources_config.json"
        elif env == "testing":
            return self.base_dir / "testing" / "data_sources_config.json"
        else:
            return self.base_dir / "data_sources_config.json"

    def ensure_backup(self, source_file):
        """确保源文件有备份"""
        if not source_file.exists():
            return None

        self.backup_dir.mkdir(exist_ok=True)

        # 创建带时间戳的备份
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"data_sources_config_{timestamp}.json"

        shutil.copy2(source_file, backup_file)
        logger.info(f"创建备份: {backup_file}")
        return backup_file

    def validate_config_data(self, data):
        """验证配置数据的有效性"""
        if not isinstance(data, list):
            raise ValueError("配置数据必须是数组格式")

        required_fields = ["id", "name", "type", "url", "enabled"]
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("配置项必须是对象格式")

            for field in required_fields:
                if field not in item:
                    raise ValueError(f"配置项缺少必需字段: {field}")

        logger.info(f"配置数据验证通过，共 {len(data)} 个数据源")
        return True

    def migrate_config(self, env):
        """执行配置迁移"""
        logger.info("开始数据源配置迁移...")

        # 检查源文件是否存在
        if not self.old_config.exists():
            logger.info("源配置文件不存在，跳过迁移")
            return False

        try:
            # 读取源配置
            with open(self.old_config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 验证数据
            self.validate_config_data(config_data)

            # 确定目标路径
            target_path = self.get_target_config_path(env)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 检查目标是否已存在
            if target_path.exists():
                logger.warning(f"目标配置文件已存在: {target_path}")
                # 如果是生产环境，不覆盖现有配置
                if env == "production":
                    logger.error("生产环境：拒绝覆盖现有配置")
                    return False

            # 创建备份
            backup_file = self.ensure_backup(self.old_config)

            # 执行迁移
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            logger.info(f"配置迁移成功: {self.old_config} -> {target_path}")
            logger.info(f"备份文件: {backup_file}")

            # 在生产环境中，询问是否删除旧文件
            if env == "production":
                response = input("是否删除旧的配置文件? (y/N): ")
                if response.lower() == 'y':
                    self.old_config.unlink()
                    logger.info("已删除旧配置文件")
                else:
                    logger.info("保留旧配置文件作为额外备份")

            return True

        except Exception as e:
            logger.error(f"配置迁移失败: {e}")
            return False

    def create_environment_readme(self):
        """创建环境说明文档"""
        readme_content = """
# RQA2025 数据源配置环境说明

## 环境隔离策略

为了避免开发/测试数据意外覆盖生产配置，系统采用环境隔离策略：

### 目录结构
```
data/
├── data_sources_config.json          # 开发环境配置
├── testing/
│   └── data_sources_config.json      # 测试环境配置
├── production/
│   └── data_sources_config.json      # 生产环境配置
└── backups/                          # 自动备份目录
    └── data_sources_config_*.json
```

### 环境配置
- **开发环境** (`RQA_ENV=development`): 使用 `data/data_sources_config.json`
- **测试环境** (`RQA_ENV=testing`): 使用 `data/testing/data_sources_config.json`
- **生产环境** (`RQA_ENV=production`): 使用 `data/production/data_sources_config.json`

### 安全保护机制

1. **生产环境保护**: 生产环境绝对不会用默认数据覆盖现有配置
2. **自动备份**: 每次保存配置时自动创建带时间戳的备份
3. **数据验证**: 保存前验证配置数据的完整性
4. **环境检测**: 根据环境变量自动选择合适的配置路径

### 迁移说明

如果您从旧版本升级：

1. 运行 `python scripts/migrate_data_sources.py` 进行安全迁移
2. 系统会自动检测环境并将配置迁移到合适的位置
3. 生产环境会要求确认是否删除旧文件

### 注意事项

- 生产环境的配置修改需要谨慎
- 定期检查备份文件的完整性
- 如需恢复配置，可从备份目录选择合适的文件
"""

        readme_path = self.base_dir / "CONFIG_ENV_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content.strip())

        logger.info(f"创建环境说明文档: {readme_path}")

def main():
    """主函数"""
    print("RQA2025 数据源配置迁移工具")
    print("=" * 50)

    migrator = DataSourceMigrator()
    env = migrator.detect_environment()

    # 执行迁移
    success = migrator.migrate_config(env)

    if success:
        # 创建说明文档
        migrator.create_environment_readme()
        print("\n✅ 迁移完成！请查看 data/CONFIG_ENV_README.md 了解详情")
    else:
        print("\n❌ 迁移失败或跳过")

    # 显示当前配置路径
    target_path = migrator.get_target_config_path(env)
    print(f"\n当前环境 ({env}) 的配置路径: {target_path}")

    if target_path.exists():
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"配置状态: ✅ 存在 {len(data)} 个数据源")
        except Exception as e:
            print(f"配置状态: ❌ 读取失败 ({e})")
    else:
        print("配置状态: ⚠️ 不存在")

if __name__ == "__main__":
    main()
