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