#!/usr/bin/env python3
"""
基础设施层配置管理公共Mixin创建工具
"""

import os
import re


def analyze_init_patterns():
    """分析__init__方法的重复模式"""

    print('=== 🔍 Phase 1.2: 分析__init__方法重复模式 ===')
    print()

    config_dir = 'src/infrastructure/config'
    init_methods = []

    # 收集所有__init__方法
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找所有__init__方法
                    init_matches = re.findall(
                        r'def __init__\([^)]*\):(.*?)(?=\n\s*def|\n\s*@|\n\s*class|\n\s*#|\nclass|\Z)', content, re.DOTALL)

                    for i, init_body in enumerate(init_matches):
                        rel_path = os.path.relpath(file_path, config_dir)
                        init_methods.append({
                            'file': f'{rel_path}#{i}',
                            'body': init_body.strip(),
                            'lines': len(init_body.strip().split('\n'))
                        })

                except Exception as e:
                    print(f'❌ 读取失败 {file}: {e}')

    print(f'📊 收集到 {len(init_methods)} 个__init__方法')
    print()

    # 分析具体模式
    pattern_stats = {
        'threading.RLock': 0,
        'threading.Lock': 0,
        'self._config': 0,
        'self._metrics': 0,
        'self._alerts': 0,
        'self._history': 0,
        'self._data': 0,
        'super().__init__': 0
    }

    for method in init_methods:
        body = method['body']
        for pattern in pattern_stats.keys():
            if pattern in body:
                pattern_stats[pattern] += 1

    print('📈 模式出现频率统计:')
    for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
        print(f'   {pattern}: {count} 次')

    print()

    # 分析可以标准化的模式
    print('🎯 可标准化的初始化模式:')

    # 1. 线程安全模式
    threading_pattern = [m for m in init_methods if 'threading.' in m['body']]
    print(f'1. 线程安全模式 ({len(threading_pattern)}个):')
    if threading_pattern:
        print('   典型模式: self._lock = threading.RLock()')
        print('   标准化为: self._lock = threading.RLock()')

    # 2. 配置存储模式
    config_pattern = [m for m in init_methods if '_config' in m['body']
                      and ('{}' in m['body'] or 'dict()' in m['body'])]
    print(f'\n2. 配置存储模式 ({len(config_pattern)}个):')
    if config_pattern:
        print('   典型模式: self._config = config or {}')
        print('   标准化为: self._config = config or {}')

    # 3. 数据收集模式
    data_pattern = [m for m in init_methods if any(
        x in m['body'] for x in ['_data', '_metrics', '_alerts', '_history'])]
    print(f'\n3. 数据收集模式 ({len(data_pattern)}个):')
    if data_pattern:
        print('   典型模式: self._data = {}, self._metrics = [], self._alerts = [], self._history = []')
        print('   标准化为: 对应的数据结构初始化')

    return pattern_stats


def create_common_mixins():
    """创建公共Mixin类"""

    print('\n🚀 创建ConfigComponentMixin类...')

    mixin_content = '''# ==================== 公共Mixin类 ====================

class ConfigComponentMixin:
    """配置组件基础Mixin类

    提供通用的初始化和基础功能，避免重复代码。
    """

    def _init_threading_support(self):
        """初始化线程安全支持"""
        from infrastructure.config.core.imports import threading
        self._lock = threading.RLock()

    def _init_config_storage(self, config=None):
        """初始化配置存储"""
        self._config = config or {}

    def _init_metrics_collection(self):
        """初始化指标收集"""
        self._metrics = {}

    def _init_alert_system(self):
        """初始化告警系统"""
        self._alerts = []

    def _init_history_tracking(self):
        """初始化历史跟踪"""
        self._history = []

    def _init_data_structures(self):
        """初始化数据结构"""
        self._data = {}

    def _init_component_attributes(self,
                                 enable_threading=True,
                                 enable_config=True,
                                 enable_metrics=False,
                                 enable_alerts=False,
                                 enable_history=False,
                                 enable_data=False,
                                 config=None):
        """统一初始化组件属性

        Args:
            enable_threading: 是否启用线程安全
            enable_config: 是否启用配置存储
            enable_metrics: 是否启用指标收集
            enable_alerts: 是否启用告警系统
            enable_history: 是否启用历史跟踪
            enable_data: 是否启用通用数据结构
            config: 初始配置数据
        """
        if enable_threading:
            self._init_threading_support()

        if enable_config:
            self._init_config_storage(config)

        if enable_metrics:
            self._init_metrics_collection()

        if enable_alerts:
            self._init_alert_system()

        if enable_history:
            self._init_history_tracking()

        if enable_data:
            self._init_data_structures()


class MonitoringMixin(ConfigComponentMixin):
    """监控组件Mixin类"""

    def __init__(self, enable_metrics=True, enable_alerts=True, enable_history=True):
        """初始化监控组件"""
        super().__init__()
        self._init_component_attributes(
            enable_threading=True,
            enable_config=True,
            enable_metrics=enable_metrics,
            enable_alerts=enable_alerts,
            enable_history=enable_history
        )

    def record_metric(self, name: str, value, timestamp=None):
        """记录指标"""
        if not hasattr(self, '_metrics'):
            self._init_metrics_collection()

        if timestamp is None:
            from infrastructure.config.core.imports import time
            timestamp = time.time()

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })

        # 限制历史记录数量
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-500:]

    def get_latest_metric(self, name: str):
        """获取最新指标"""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1]
        return None


class CRUDOperationsMixin(ConfigComponentMixin):
    """CRUD操作Mixin类"""

    def __init__(self):
        """初始化CRUD操作组件"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_config=True)

    def create(self, key: str, value):
        """创建记录"""
        with self._lock:
            self._config[key] = value
            self._record_operation('create', key, value)

    def read(self, key: str):
        """读取记录"""
        with self._lock:
            return self._config.get(key)

    def update(self, key: str, value):
        """更新记录"""
        with self._lock:
            if key in self._config:
                old_value = self._config[key]
                self._config[key] = value
                self._record_operation('update', key, value, old_value)
                return True
            return False

    def delete(self, key: str):
        """删除记录"""
        with self._lock:
            if key in self._config:
                value = self._config[key]
                del self._config[key]
                self._record_operation('delete', key, value)
                return True
            return False

    def _record_operation(self, operation: str, key: str, value=None, old_value=None):
        """记录操作历史"""
        if not hasattr(self, '_history'):
            self._init_history_tracking()

        from infrastructure.config.core.imports import time
        record = {
            'operation': operation,
            'key': key,
            'value': value,
            'old_value': old_value,
            'timestamp': time.time()
        }

        self._history.append(record)

        # 限制历史记录数量
        if len(self._history) > 1000:
            self._history = self._history[-500:]


class ComponentLifecycleMixin(ConfigComponentMixin):
    """组件生命周期Mixin类"""

    def __init__(self):
        """初始化生命周期管理"""
        super().__init__()
        self._init_component_attributes(enable_threading=True)
        self._initialized = False
        self._started = False
        self._stopped = False

    def initialize(self):
        """初始化组件"""
        if not self._initialized:
            self._do_initialize()
            self._initialized = True

    def start(self):
        """启动组件"""
        if not self._started:
            if not self._initialized:
                self.initialize()
            self._do_start()
            self._started = True

    def stop(self):
        """停止组件"""
        if self._started and not self._stopped:
            self._do_stop()
            self._stopped = True

    def restart(self):
        """重启组件"""
        self.stop()
        self._started = False
        self._stopped = False
        self.start()

    def _do_initialize(self):
        """子类实现具体的初始化逻辑"""
        pass

    def _do_start(self):
        """子类实现具体的启动逻辑"""
        pass

    def _do_stop(self):
        """子类实现具体的停止逻辑"""
        pass

    @property
    def is_initialized(self):
        """检查是否已初始化"""
        return self._initialized

    @property
    def is_started(self):
        """检查是否已启动"""
        return self._started

    @property
    def is_stopped(self):
        """检查是否已停止"""
        return self._stopped


# ==================== 向后兼容性 ====================

# 为现有类提供别名，确保向后兼容
ConfigComponentMixinAlias = ConfigComponentMixin
MonitoringMixinAlias = MonitoringMixin
CRUDOperationsMixinAlias = CRUDOperationsMixin
ComponentLifecycleMixinAlias = ComponentLifecycleMixin
'''

    # 写入公共Mixin文件
    mixin_file = 'src/infrastructure/config/core/common_mixins.py'
    os.makedirs(os.path.dirname(mixin_file), exist_ok=True)

    with open(mixin_file, 'w', encoding='utf-8') as f:
        f.write(mixin_content)

    print(f'✅ 已创建公共Mixin文件: {mixin_file}')

    return mixin_file


if __name__ == '__main__':
    # 分析模式
    pattern_stats = analyze_init_patterns()

    # 创建Mixin
    mixin_file = create_common_mixins()

    print(f'\n🎯 Phase 1.2 完成！')
    print(f'已创建公共Mixin类，共提取了以下模式:')
    print(f'   - 线程安全支持')
    print(f'   - 配置存储')
    print(f'   - 指标收集')
    print(f'   - 告警系统')
    print(f'   - 历史跟踪')
    print(f'   - CRUD操作')
    print(f'   - 组件生命周期')
