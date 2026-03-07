import os
import json
import re
from datetime import datetime, timedelta
from jinja2 import Template


def safe_load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取{path}失败，使用默认值。原因：{e}")
        return default


def parse_low_coverage(coverage_path, threshold=80):
    low_items = []
    if not os.path.exists(coverage_path):
        print(f"未找到覆盖率报告：{coverage_path}")
        return low_items
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'utf-16', 'latin1']
    for enc in encodings:
        try:
            with open(coverage_path, 'r', encoding=enc) as f:
                for line in f:
                    m = re.match(r'(\S+\.py)\s+(\d+)%', line)
                    if m:
                        fname, cov = m.group(1), int(m.group(2))
                        if cov < threshold:
                            low_items.append({'file': fname, 'coverage': cov})
            return low_items
        except UnicodeDecodeError:
            continue
    print(f"无法识别{coverage_path}的编码格式，请手动转为utf-8或gbk！")
    return low_items


def generate_suggestions(low_coverage_items, owner="待分配", days=7):
    suggestions = []
    now = datetime.now()
    for idx, item in enumerate(sorted(low_coverage_items, key=lambda x: x['coverage'])):
        priority = "高" if item['coverage'] < 50 else "中" if item['coverage'] < 70 else "低"
        deadline = (now.replace(hour=0, minute=0, second=0, microsecond=0) +
                    timedelta(days=days)).strftime('%Y-%m-%d')
        suggestions.append({
            "file": item['file'],
            "coverage": item['coverage'],
            "priority": priority,
            "owner": owner,
            "suggestion": f"补充单测，提升覆盖率至80%以上",
            "deadline": deadline
        })
    return suggestions


def merge_tech_debt(tech_debt, suggestions):
    auto_items = []
    for s in suggestions:
        auto_items.append({
            "module": s["file"],
            "desc": f"覆盖率仅{s['coverage']}%，需补充单测（自动发现）",
            "priority": s["priority"],
            "owner": s["owner"],
            "suggestion": s["suggestion"],
            "deadline": s["deadline"]
        })
    merged = {"items": tech_debt.get("items", []) + auto_items,
              "summary": tech_debt.get("summary", "")}
    return merged


# 读取数据
progress = safe_load_json('docs/progress_tracking.json', {"layers": [
], "overall_completion": "0%", "overall_coverage": "0%", "expected_finish": "-", "summary": ""})
tech_debt = safe_load_json('docs/technical_debt.json', {"items": [], "summary": ""})
low_coverage_items = parse_low_coverage('docs/coverage_report_latest.txt', threshold=80)
suggestions = generate_suggestions(low_coverage_items, owner="待分配", days=7)
merged_tech_debt = merge_tech_debt(tech_debt, suggestions)

# Jinja2模板
progress_tpl = """
# RQA2025 自动化测试全局进度报告

生成时间：{{ now }}

## 各层完成情况

| 层级 | 完成度 | 覆盖率 | 通过数 | 失败数 | 跳过数 | 优先级 | 最新进展 |
|------|--------|--------|--------|--------|--------|--------|----------|
{% for layer in progress['layers'] -%}
| {{ layer['name'] }} | {{ layer['completion'] }} | {{ layer['coverage'] }} | {{ layer['passed'] }} | {{ layer['failed'] }} | {{ layer['skipped'] }} | {{ layer['priority'] }} | {{ layer['latest'] }} |
{% endfor %}

## 低覆盖点明细（低于{{ threshold }}%）

{% if low_coverage_items %}
| 文件 | 覆盖率 |
|------|--------|
{% for item in low_coverage_items -%}
| {{ item.file }} | {{ item.coverage }}% |
{% endfor %}
{% else %}
无低覆盖点，整体质量良好！
{% endif %}

## 低覆盖点补测建议

{% if suggestions %}
| 文件 | 当前覆盖率 | 优先级 | 责任人 | 建议 | 预计完成时间 |
|------|------------|--------|--------|------|--------------|
{% for s in suggestions -%}
| {{ s.file }} | {{ s.coverage }}% | {{ s.priority }} | {{ s.owner }} | {{ s.suggestion }} | {{ s.deadline }} |
{% endfor %}
{% else %}
无低覆盖点，无需补测建议。
{% endif %}

## 总体统计

- 平均完成度：{{ progress['overall_completion'] }}
- 平均覆盖率：{{ progress['overall_coverage'] }}
- 预计交付日期：{{ progress['expected_finish'] }}

## 最新进展与优化建议

{{ progress['summary'] }}
"""

tech_debt_tpl = """
# RQA2025 技术债务与优化建议报告

生成时间：{{ now }}

## 主要技术债务项（含自动发现的低覆盖点补测建议）

| 序号 | 模块/层 | 问题描述 | 优先级 | 责任人 | 解决建议 | 预计完成时间 |
|------|---------|----------|--------|--------|----------|--------------|
{% for item in tech_debt['items'] -%}
| {{ loop.index }} | {{ item['module'] }} | {{ item['desc'] }} | {{ item['priority'] }} | {{ item['owner'] }} | {{ item['suggestion'] }} | {{ item['deadline'] }} |
{% endfor %}

## 优化建议

{{ tech_debt['summary'] }}
"""

# 渲染
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
progress_md = Template(progress_tpl).render(
    progress=progress,
    now=now,
    low_coverage_items=low_coverage_items,
    threshold=80,
    suggestions=suggestions
)
tech_debt_md = Template(tech_debt_tpl).render(tech_debt=merged_tech_debt, now=now)

# 历史归档
dest_dir = 'docs/archive'
os.makedirs(dest_dir, exist_ok=True)
for fname in ['progress_report.md', 'technical_debt_report.md']:
    fpath = f'docs/{fname}'
    if os.path.exists(fpath):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.rename(fpath, f'{dest_dir}/{fname}.{ts}')

# 写入新报告
with open('docs/progress_report.md', 'w', encoding='utf-8') as f:
    f.write(progress_md)
with open('docs/technical_debt_report.md', 'w', encoding='utf-8') as f:
    f.write(tech_debt_md)

print("全局进度与技术债务报告已自动生成，低覆盖点补测建议已同步到技术债务报告！")
