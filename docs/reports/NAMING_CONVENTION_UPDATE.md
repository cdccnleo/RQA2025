# RQA2025 报告命名规范调整总结

## 📋 调整概览

根据项目需求，对报告命名规范进行了重要调整，确保报告始终代表最新版本，避免文件名中包含日期和版本信息。

## 🔄 主要变更

### 1. 基本命名格式调整

**调整前**:
```
{category}_{type}_{subject}_{date}_{version}.{extension}
```

**调整后**:
```
{category}_{type}_{subject}.{extension}
```

### 2. 移除的组件

- **date**: 日期格式（YYYYMMDD）
- **version**: 版本标识（v1, v2, final）

### 3. 保留的组件

- **category**: 报告类别（project, technical, business, operational, research）
- **type**: 报告类型（progress, completion, analysis, test, audit）
- **subject**: 报告主题（deployment, performance, security）
- **extension**: 文件扩展名（.md, .json, .html）

## 🎯 调整原因

### 1. 时效性考虑
- 报告具有时效性，应该始终代表最新状态
- 避免文件名中的日期信息造成混淆
- 确保用户总是访问到最新版本

### 2. 版本控制方式
- 版本控制通过报告内容实现，而非文件名
- 在报告内容中记录版本信息和更新时间
- 使用报告元数据管理版本历史

### 3. 简化管理
- 减少文件名的复杂性
- 提高文件的可读性和可维护性
- 避免版本冲突和重复文件

## 📊 实施结果

### 文件重命名统计
- **总重命名文件数**: 242个
- **成功重命名**: 240个
- **失败重命名**: 2个（文件名冲突）
- **处理时间**: 约30秒

### 重命名示例

**调整前**:
```
project_progress_deployment_20250727_v1.md
technical_test_performance_20250727.json
business_analysis_trading_20250727.md
```

**调整后**:
```
project_progress_deployment.md
technical_test_performance.json
business_analysis_trading.md
```

## 📝 更新的文档

### 1. 命名规范文档
- **文件**: `docs/reports/REPORT_NAMING_STANDARDS.md`
- **更新内容**: 基本格式、命名示例、特殊规则

### 2. 快速参考指南
- **文件**: `docs/reports/QUICK_REFERENCE.md`
- **更新内容**: 命名规范提醒、查找方式

### 3. 文档索引
- **文件**: `docs/DOCUMENT_INDEX.md`
- **更新内容**: 报告命名规范说明

## 🔧 技术实现

### 1. 重命名脚本
- **文件**: `scripts/reports/rename_reports.py`
- **功能**: 自动清理文件名中的日期和版本信息
- **处理模式**: 正则表达式匹配和替换

### 2. 清理规则
```python
patterns = [
    r'_\d{8}',  # YYYYMMDD
    r'_\d{8}_\d{6}',  # YYYYMMDD_HHMMSS
    r'_\d{8}_\d{4}',  # YYYYMMDD_HHMM
    r'_v\d+',  # v1, v2, v3...
    r'_final',  # final
    r'_draft',  # draft
    r'_review',  # review
    r'_\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
    r'_\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
]
```

## 🚀 最佳实践

### 1. 创建新报告
- 使用简化的命名格式
- 在报告内容中记录版本信息
- 使用元数据管理时间戳

### 2. 维护现有报告
- 定期更新报告内容
- 在内容中维护版本历史
- 使用内容更新时间判断最新版本

### 3. 查找报告
- 按修改时间排序
- 使用文件内容中的时间信息
- 通过目录结构快速定位

## 📋 检查清单

### 创建新报告时
- [ ] 使用新的命名格式（无日期版本）
- [ ] 在报告内容中记录版本信息
- [ ] 包含必要的元数据
- [ ] 遵循模板规范

### 维护现有报告时
- [ ] 检查命名一致性
- [ ] 更新报告内容中的版本信息
- [ ] 维护内容更新时间
- [ ] 清理过时信息

### 查找报告时
- [ ] 使用修改时间排序
- [ ] 查看报告内容中的版本信息
- [ ] 通过目录结构定位
- [ ] 使用搜索功能

## 🔗 相关文档

- [报告命名规范](REPORT_NAMING_STANDARDS.md)
- [快速参考指南](QUICK_REFERENCE.md)
- [报告组织规范](../README.md)
- [文档索引](../DOCUMENT_INDEX.md)

---

**调整日期**: 2025-01-27  
**维护者**: 项目团队  
**状态**: ✅ 已完成调整 