import re

# 读取文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复断言
content = re.sub(r'assert exc\.field == "email"', 'assert exc.context["field"] == "email"', content)
content = re.sub(r'assert exc\.value == "invalid-email"', 'assert exc.context["value"] == "invalid-email"', content)

# 写入文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed ValidationError test assertions')

