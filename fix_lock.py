# 直接修复EventSubscriber
with open('src/core/event_bus/components/event_subscriber.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有的with self._lock为acquire/release模式
content = content.replace('        with self._lock:', '''        self._lock.acquire()
        try:''')

content = content.replace('        logger.debug(f"订阅事件: {event_type_str}")', '''        finally:
            self._lock.release()

        logger.debug(f"订阅事件: {event_type_str}")''')

content = content.replace('        return handlers, async_handlers', '''        finally:
            self._lock.release()
        return handlers, async_handlers''')

content = content.replace('            return sync_count + async_count', '''        finally:
            self._lock.release()
        return sync_count + async_count''')

with open('src/core/event_bus/components/event_subscriber.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed EventSubscriber lock mechanism')

