#!/usr/bin/env python3
"""
移除所有硬编码的删除按钮
"""

def remove_hardcoded_delete_buttons():
    """移除HTML中所有硬编码的deleteDataSource按钮"""

    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 移除所有硬编码的删除按钮（不包含${source.id}的）
    import re
    original_content = content
    content = re.sub(
        r'\s*<button onclick="deleteDataSource\(\'[^$][^\']+\'\)" class="text-red-600 hover:text-red-900[^"]*">\s*<i class="fas fa-trash"></i>\s*删除\s*</button>',
        '',
        content
    )

    if content != original_content:
        with open('web-static/data-sources-config.html', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 已移除所有硬编码的删除按钮")
        return True
    else:
        print("ℹ️ 没有找到需要移除的硬编码删除按钮")
        return False

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    remove_hardcoded_delete_buttons()
