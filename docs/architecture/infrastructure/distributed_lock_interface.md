# 分布式锁接口设计

## 1. 目标
- 提供统一的分布式锁接口，支持多种后端（Redis、etcd、ZooKeeper）
- 支持阻塞/非阻塞获取、超时、自动续约、自动释放

## 2. 接口定义（Python示例）

```python
class DistributedLock:
    def acquire(self, key: str, timeout: float = None, blocking: bool = True) -> bool:
        """获取分布式锁"""
        pass

    def release(self, key: str) -> bool:
        """释放分布式锁"""
        pass

    def is_locked(self, key: str) -> bool:
        """判断锁是否被持有"""
        pass

    def renew(self, key: str, ttl: float) -> bool:
        """续约锁"""
        pass
```

## 3. Redis Redlock实现要点
- 多节点投票机制
- 锁唯一ID与过期时间
- 自动续约与失效检测

## 4. etcd实现要点
- 利用etcd的lease机制
- 支持watch变更事件

## 5. 兼容性与扩展性
- 统一接口，便于后端切换
- 支持插件化扩展

## 6. 测试用例建议
- 多进程/多线程并发获取与释放
- 超时与自动释放
- 网络分区与节点失效场景