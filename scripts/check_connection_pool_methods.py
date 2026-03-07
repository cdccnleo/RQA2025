#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查ConnectionPool类的方法"""

import ast
import sys
sys.path.insert(0, 'src')

# 读取文件内容
with open('src/infrastructure/utils/components/connection_pool.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 解析AST
tree = ast.parse(content)

# 查找ConnectionPool类
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'ConnectionPool':
        print(f"找到ConnectionPool类，行号: {node.lineno}")
        print("类的方法:")
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # 计算缩进
                lines_before = content[:content.find(item.name, content.split('\n')[:item.lineno-1].join('\n').__len__())].split('\n')
                indent = len(lines_before[-1]) - len(lines_before[-1].lstrip())
                print(f"  - {item.name} (行号: {item.lineno}, 缩进: {indent})")

# 直接搜索文本
print("\n文本搜索:")
import re
matches = list(re.finditer(r'(\s*)def (get_connection|put_connection)', content))
for match in matches:
    indent_len = len(match.group(1))
    method_name = match.group(2)
    line_num = content[:match.start()].count('\n') + 1
    print(f"  - {method_name} (行号: {line_num}, 缩进: {indent_len}空格)")

