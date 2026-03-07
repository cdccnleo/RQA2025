# 读取文件
with open('src/core/event_bus/persistence/event_persistence.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复缩进问题
if len(lines) > 566:  # 第567行 (0索引是566)
    lines[566] = '            logger.error(f"按参数存储事件失败 {event_id}: {e}")\n'
if len(lines) > 567:  # 第568行 (0索引是567)
    lines[567] = '            return False\n'

# 写回文件
with open('src/core/event_bus/persistence/event_persistence.py', 'w', encoding='utf-8') as f:
    f.write(''.join(lines))

print('Fixed exception block indentation')

