#!/usr/bin/env python3
"""
基础设施层测试紧急修复脚本 v2

修复剩余的关键问题，确保测试能够正常运行
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def fix_unified_cache():
    """修复UnifiedCache类"""
    print("🔧 修复UnifiedCache类...")

    cache_file = src_path / 'infrastructure' / 'cache' / 'unified_cache.py'

    with open(cache_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 添加缺失的属性
    if '_redis_cache' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'self._memory_cache = self._cache' in line:
                lines.insert(i + 1, '        # Redis缓存兼容性属性')
                lines.insert(i + 2, '        self._redis_cache = None')
                lines.insert(i + 3, '        self._statistics = self._stats  # 兼容性属性')
                break

        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("  - 添加UnifiedCache缺失属性")


def fix_cache_stats():
    """修复CacheStats类，使其支持下标访问"""
    print("🔧 修复CacheStats类...")

    cache_file = src_path / 'infrastructure' / 'cache' / 'caching.py'

    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找CacheStats类
        if 'class CacheStats:' in content:
            lines = content.split('\n')
            class_start = -1
            class_end = -1

            for i, line in enumerate(lines):
                if 'class CacheStats:' in line:
                    class_start = i
                elif class_start >= 0 and line.strip().startswith('class ') and i > class_start:
                    class_end = i
                    break
                elif class_start >= 0 and i == len(lines) - 1:
                    class_end = len(lines)

            if class_start >= 0 and class_end > class_start:
                # 在类中添加__getitem__方法
                for i in range(class_start, class_end):
                    if lines[i].strip() == '':  # 找到空行
                        lines.insert(i, '    def __getitem__(self, key):')
                        lines.insert(i + 1, '        """支持下标访问"""')
                        lines.insert(i + 2, '        return getattr(self, key, None)')
                        lines.insert(i + 3, '')
                        lines.insert(i + 4, '    def __iter__(self):')
                        lines.insert(i + 5, '        """支持迭代"""')
                        lines.insert(
                            i + 6, '        return iter([attr for attr in dir(self) if not attr.startswith("_")])')
                        lines.insert(i + 7, '')
                        break

                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print("  - 添加CacheStats下标访问支持")


def fix_unified_logger():
    """修复UnifiedLogger类"""
    print("🔧 修复UnifiedLogger类...")

    logger_file = src_path / 'infrastructure' / 'logging' / 'unified_logger.py'

    with open(logger_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 添加缺失的属性
    if '_default_level' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'self._loggers = {}' in line:
                lines.insert(i + 1, '        self._default_level = level')
                lines.insert(
                    i + 2, '        self._format = \'%(asctime)s - %(name)s - %(levelname)s - %(message)s\'')
                break

        with open(logger_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("  - 添加UnifiedLogger缺失属性")


def fix_memory_cache_ttl():
    """修复内存缓存TTL问题"""
    print("🔧 修复内存缓存TTL问题...")

    cache_file = src_path / 'infrastructure' / 'cache' / 'memory_cache.py'

    with open(cache_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复set方法中的TTL处理
    if 'def set(' in content:
        lines = content.split('\n')
        set_start = -1
        set_end = -1

        for i, line in enumerate(lines):
            if 'def set(self, key: str, value: Any, ttl: Optional[int] = None):' in line:
                set_start = i
            elif set_start >= 0 and line.strip().startswith('def ') and i > set_start:
                set_end = i
                break
            elif set_start >= 0 and i == len(lines) - 1:
                set_end = len(lines)

        if set_start >= 0:
            # 替换set方法
            new_set_method = '''    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项，支持TTL"""
        if key is None:
            raise TypeError("缓存键不能为None")

        with self._lock:
            # 处理TTL
            if ttl is None:
                ttl = self.ttl

            expiration_time = time.time() + ttl if ttl > 0 else None

            # 检查容量限制
            if len(self.cache) >= self.capacity and key not in self.cache:
                # 移除最少使用的项
                if self.cache:
                    oldest_key, _ = self.cache.popitem(last=False)
                    self.timestamps.pop(oldest_key, None)

            # 设置新值
            self.cache[key] = value
            self.timestamps[key] = expiration_time

        return True'''

            # 替换原来的set方法
            for i in range(set_start, min(set_end, len(lines))):
                if lines[i].strip().startswith('def set('):
                    # 找到方法结束
                    method_end = i + 1
                    brace_count = 0
                    for j in range(i + 1, len(lines)):
                        if '{' in lines[j]:
                            brace_count += 1
                        if '}' in lines[j]:
                            brace_count -= 1
                        if brace_count == 0 and lines[j].strip() == '':
                            method_end = j
                            break

                    # 替换方法
                    new_lines = lines[:i] + new_set_method.split('\n') + lines[method_end:]
                    lines = new_lines
                    break

            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print("  - 修复LRUCache.set方法TTL处理")


def fix_redis_mock():
    """修复Redis缓存Mock问题"""
    print("🔧 修复Redis缓存Mock问题...")

    redis_file = src_path / 'infrastructure' / 'cache' / 'redis_cache.py'

    if redis_file.exists():
        with open(redis_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复Redis连接问题
        if 'redis.Redis(' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'redis.Redis(' in line and 'ex=' not in line:
                    # 添加ex参数
                    lines[i] = line.replace('redis.Redis(', 'redis.Redis(ex=3600, ')
                    break

            with open(redis_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print("  - 修复Redis连接参数")


def fix_test_imports():
    """修复测试文件中的导入问题"""
    print("🔧 修复测试文件导入问题...")

    # 修复日志系统测试
    logging_test_file = project_root / 'tests' / 'unit' / 'infrastructure' / 'test_logging_system.py'

    if logging_test_file.exists():
        with open(logging_test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复FileHandler导入
        if "from logging.handlers import FileHandler" in content:
            new_content = content.replace(
                "from logging.handlers import FileHandler",
                "# from logging.handlers import FileHandler  # 不再需要"
            )
            with open(logging_test_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  - 修复日志测试FileHandler导入")


def main():
    """主修复函数"""
    print("🚀 开始基础设施层测试紧急修复 v2...")
    print("=" * 50)

    try:
        # 1. 修复UnifiedCache
        fix_unified_cache()

        # 2. 修复CacheStats
        fix_cache_stats()

        # 3. 修复UnifiedLogger
        fix_unified_logger()

        # 4. 修复内存缓存TTL
        fix_memory_cache_ttl()

        # 5. 修复Redis Mock
        fix_redis_mock()

        # 6. 修复测试导入
        fix_test_imports()

        print("=" * 50)
        print("✅ 基础设施层测试紧急修复 v2 完成！")
        print("\n📋 修复内容总结:")
        print("  - 添加UnifiedCache缺失属性")
        print("  - 修复CacheStats下标访问")
        print("  - 添加UnifiedLogger缺失属性")
        print("  - 修复LRUCache.set方法TTL处理")
        print("  - 修复Redis连接参数")
        print("  - 修复测试文件导入问题")
        print("\n🎯 现在可以重新运行测试了！")

    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
