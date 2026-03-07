# 通用工具迁移报告

## 📊 迁移概况

- **迁移时间**: 20250831_173604
- **备份位置**: `C:\PythonProject\RQA2025\backups\common_tools_migration_20250831_173604`

## 🎨 可视化工具迁移

- **迁移文件数**: 1
- **跳过文件数**: 0
- **错误数**: 0

### 迁移详情
- `src/backtest/visualization.py` → `src/core/visualization.py`
- `src/backtest/visualizer.py` → `src/core/visualizer.py`

## 🔧 工具函数迁移

- **迁移文件数**: 1
- **跳过文件数**: 0
- **错误数**: 0

### 迁移详情
- `src/strategy/backtest/utils/backtest_utils.py` → `src/utils/backtest_utils.py`

## 🔄 导入更新

- **更新文件数**: 0
- **错误数**: 0

## 📦 模块创建

- **可视化模块**: ❌ 创建失败
- **工具模块**: ✅ 更新成功

## 🗂️ 保留文件

以下文件因其专用性而保留在原位置：

### src/backtest/ 保留文件
- `config_manager.py` - 回测专用配置管理器
- `data_loader.py` - 回测专用数据加载器
- `__init__.py` - 包结构文件
- 各子目录的 `__init__.py`

## ✅ 迁移完成

通用工具已成功迁移到适当的位置：
- **可视化工具** → `src/core/`
- **工具函数** → `src/utils/`
- **专用工具** → 保留在 `src/backtest/`

---
**迁移报告生成时间**: 20250831_173604
