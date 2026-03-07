#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def fix_event_handler_calls():
    with open('tests/unit/infrastructure/events/test_event_driven_system.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复EventHandler构造函数调用
    pattern = r'handler = EventHandler\(name="test_handler"\)'
    replacement = '''async def dummy_handler(event):
            pass

        handler = EventHandler(dummy_handler)'''

    content = re.sub(pattern, replacement, content)

    with open('tests/unit/infrastructure/events/test_event_driven_system.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print('Fixed EventHandler constructor calls')

if __name__ == '__main__':
    fix_event_handler_calls()