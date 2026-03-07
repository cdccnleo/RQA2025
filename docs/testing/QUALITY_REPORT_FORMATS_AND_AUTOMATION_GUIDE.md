# RQA2025 多格式质量报告生成与自动化指南

## 1. 多格式报告生成脚本用法
- 脚本路径：`scripts/ops/generate_quality_report_formats.py`
- 支持将`reports/quality/quality_report.md`自动转为HTML和PDF格式
- 需安装依赖：`pip install markdown2 pdfkit`，PDF需本地安装`wkhtmltopdf`
- 运行命令：
  ```bash
  python scripts/ops/generate_quality_report_formats.py
  ```
- 产出：
  - `reports/quality/quality_report.html`
  - `reports/quality/quality_report.pdf`

## 2. 多格式报告推送与归档建议
- 可结合多平台同步脚本，将HTML/PDF报告一并推送到Wiki/知识平台/邮件/IM等渠道
- 建议在CI产物中归档所有格式报告，便于团队成员下载与复盘

## 3. 复盘自动化与多格式集成
- 定期自动生成多格式报告，自动同步到Wiki/知识库
- 复盘纪要可直接引用HTML/PDF报告，提升可读性与归档价值
- 支持一键归档与多渠道分发

## 4. 常见问题与FAQ
- Q: PDF报告生成失败如何排查？
  A: 检查是否已安装`wkhtmltopdf`和`pdfkit`，查看脚本输出日志。
- Q: 如何扩展更多格式？
  A: 可结合pandoc等工具扩展为DOCX、PPT等格式。
- Q: 如何集成到CI/CD？
  A: 在CI流程中调用生成脚本，归档所有格式报告。

---
如需更多多格式报告与自动化建议，请联系质量负责人。 