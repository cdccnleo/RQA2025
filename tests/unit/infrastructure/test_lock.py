#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lock 测试用例
"""

import pytest
import sys
import os
import threading
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.infrastructure import lock
from src.infrastructure.lock import LockManager, get_default_lock_manager

class TestLock:
    """测试lock模块"""
    
    def test_import(self):
        """测试模块导入"""
        assert lock is not None
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # TODO: 添加具体的测试用例
        pass
    
    def test_error_handling(self):
        """测试错误处理"""
        # TODO: 添加错误处理测试用例
        pass
    
    def test_configuration(self):
        """测试配置相关功能"""
        # TODO: 添加配置测试用例
        pass

def test_acquire_and_release():
    lm = LockManager()
    assert lm.acquire('test') is True
    assert lm.get_lock_stats()['test'] == 'locked'
    lm.release('test')
    assert lm.get_lock_stats()['test'] == 'unlocked'

def test_acquire_timeout():
    lm = LockManager()
    assert lm.acquire('t1') is True
    # 新线程尝试获取同名锁，超时应返回False
    def try_lock(result):
        result.append(lm.acquire('t1', timeout=0.1))
    result = []
    t = threading.Thread(target=try_lock, args=(result,))
    t.start()
    t.join()
    assert result == [False]
    lm.release('t1')

def test_release_unacquired():
    lm = LockManager()
    lm.acquire('t2')
    lm.release('t2')
    # 再次释放应抛异常
    with pytest.raises(RuntimeError):
        lm.release('t2')

def test_get_lock_stats_empty():
    lm = LockManager()
    assert lm.get_lock_stats() == {}

def test_singleton():
    lm1 = get_default_lock_manager()
    lm2 = get_default_lock_manager()
    assert lm1 is lm2

def test_multiple_locks():
    lm = LockManager()
    assert lm.acquire('a')
    assert lm.acquire('b')
    assert set(lm.get_lock_stats().keys()) == {'a', 'b'}
    lm.release('a')
    lm.release('b')

def test_acquire_invalid():
    lm = LockManager()
    # 空字符串作为锁名是允许的
    assert lm.acquire("")
    lm.release("")
    # 传递超长字符串
    long_name = "x" * 10000
    assert lm.acquire(long_name)
    lm.release(long_name)

def test_concurrent_acquire():
    lm = LockManager()
    results = []
    def worker():
        got = lm.acquire('c', timeout=0.2)
        results.append(got)
        if got:
            time.sleep(0.1)
            lm.release('c')
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 至少有1个线程能获取到锁
    assert results.count(True) >= 1
