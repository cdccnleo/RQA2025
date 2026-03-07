import os
import markdown2
try:
    import pdfkit
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

MD_PATH = 'reports/quality/quality_report.md'
HTML_PATH = 'reports/quality/quality_report.html'
PDF_PATH = 'reports/quality/quality_report.pdf'

if not os.path.exists(MD_PATH):
    print(f'[ERROR] Markdown质量报告不存在: {MD_PATH}')
    exit(1)

with open(MD_PATH, 'r', encoding='utf-8') as f:
    md_content = f.read()

# 生成HTML
html_content = markdown2.markdown(md_content)
with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f'[OK] 已生成HTML报告: {HTML_PATH}')

# 生成PDF（需安装wkhtmltopdf和pdfkit）
if PDF_SUPPORT:
    try:
        pdfkit.from_string(html_content, PDF_PATH)
        print(f'[OK] 已生成PDF报告: {PDF_PATH}')
    except Exception as e:
        print(f'[ERROR] 生成PDF报告失败: {e}')
else:
    print('[SKIP] 未安装pdfkit，跳过PDF报告生成')
