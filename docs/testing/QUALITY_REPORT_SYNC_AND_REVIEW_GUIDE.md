# RQA2025 质量报告自动同步与复盘指南

## 1. 自动同步脚本用法
- 脚本路径：`scripts/ops/sync_quality_report_to_wiki.py`
- 支持将`reports/quality/quality_report.md`自动推送到多个Wiki/知识平台（如WIKI、NOTION、语雀等）
- 需配置环境变量：
  - `WIKI_API_URL`、`WIKI_API_TOKEN`
  - `NOTION_API_URL`、`NOTION_API_TOKEN`
  - `YUQUE_API_URL`、`YUQUE_API_TOKEN`
- 运行命令：
  ```bash
  export WIKI_API_URL=... WIKI_API_TOKEN=... NOTION_API_URL=... NOTION_API_TOKEN=... YUQUE_API_URL=... YUQUE_API_TOKEN=...
  python scripts/ops/sync_quality_report_to_wiki.py
  ```

## 2. Wiki/知识平台集成建议
- 推荐使用支持API推送的Wiki/知识平台（如Confluence、Notion、语雀、企业自建Wiki等）
- 可结合CI定期任务自动同步，或手动触发
- 支持多格式（Markdown/HTML）报告同步

## 3. 团队定期复盘流程与模板
- 建议每周/每月定期复盘质量报告，分析短板与优化点
- 复盘模板示例：
  - 质量指标趋势（覆盖率、通过率、性能等）
  - 主要问题与改进建议
  - 责任人/跟进计划
- 复盘结论同步到Wiki/质量看板，便于团队共享

## 4. 常见问题与FAQ
- Q: API推送失败如何排查？
  A: 检查API地址、Token、网络连通性，查看脚本输出日志。
- Q: 如何支持多平台同步？
  A: 可扩展脚本，支持多API推送或Webhook。
- Q: 如何自动化集成到CI？
  A: 在CI流程最后一步调用同步脚本，确保报告已生成。

## 2. 自动化复盘建议
- 可扩展脚本，定期汇总质量报告、历史趋势、主要问题与改进建议，自动生成复盘纪要
- 支持一键归档到Wiki/知识库，便于团队复盘与经验积累
- 可结合CI定期任务自动提醒团队成员参与复盘

---
如需更多质量报告同步与复盘建议，请联系质量负责人。 