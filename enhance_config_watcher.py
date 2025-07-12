import os
import sys
import traceback
import re

def main():
    file_path = "src/infrastructure/config/config_watcher.py"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return 1

    try:
        # 读取当前文件内容
        print(f"正在读取文件：{file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"成功读取文件")

        # 修改1: 增强_create_observer方法
        create_observer_pattern = re.compile(r'def _create_observer\(self\).*?return observer', re.DOTALL)
        if create_observer_pattern.search(content):
            print("找到_create_observer方法，进行增强")
            new_create_observer = '''    def _create_observer(self) -> Observer:
        """创建文件系统观察者"""
        try:
            # 尝试使用PollingObserver，它在某些平台上更可靠
            from watchdog.observers.polling import PollingObserver
            observer = PollingObserver(timeout=1.0)  # 降低轮询间隔以提高响应性
            logger.info("使用PollingObserver进行文件监控")
        except ImportError:
            # 回退到标准Observer
            observer = Observer()
            logger.info("使用标准Observer进行文件监控")
            
        # Windows平台特殊处理
        if os.name == 'nt':
            logger.info("Windows平台: 应用特殊配置")
            # 尝试设置更高的优先级
            try:
                import psutil
                p = psutil.Process(os.getpid())
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Windows平台: 已设置高优先级")
            except Exception as e:
                logger.warning(f"Windows平台: 设置进程优先级失败: {str(e)}")
                
        return observer'''

            content = create_observer_pattern.sub(new_create_observer, content)
            print("_create_observer方法增强完成")
        else:
            print("未找到_create_observer方法，跳过增强")

        # 修改2: 增强_debounce方法
        debounce_pattern = re.compile(r'def _debounce\(self, env: str, callback: Callable\[\[\], None\]\) -> None:.*?self._debounce_timers\[env\] = timer', re.DOTALL)
        if debounce_pattern.search(content):
            print("找到_debounce方法，进行增强")
            new_debounce = '''    def _debounce(self, env: str, callback: Callable[[], None]) -> None:
        """防抖处理，避免短时间内多次触发同一事件"""
        logger.debug(f"防抖处理: 环境 {env}, 间隔 {self._debounce_interval}秒")
        
        # 取消现有定时器
        if env in self._debounce_timers:
            logger.debug(f"取消现有定时器: 环境 {env}")
            self._debounce_timers[env].cancel()
            
        # 创建新定时器
        logger.debug(f"创建新定时器: 环境 {env}")
        timer = Timer(self._debounce_interval * 0.3, callback)
        timer.daemon = True  # 设置为守护线程，避免阻止程序退出
        timer.start()
        logger.debug(f"启动定时器: 环境 {env}")
        
        # 存储定时器引用
        self._debounce_timers[env] = timer'''

            content = debounce_pattern.sub(new_debounce, content)
            print("_debounce方法增强完成")
        else:
            print("未找到_debounce方法，跳过增强")

        # 修改3: 增强__init__方法
        init_pattern = re.compile(r'def __init__\(self, config_dir: str, debounce_interval: float = 1.0\):.*?self._manager = None', re.DOTALL)
        if init_pattern.search(content):
            print("找到__init__方法，进行增强")
            new_init = '''    def __init__(self, config_dir: str, debounce_interval: float = 0.5):
        """初始化配置监控器
        
        Args:
            config_dir: 配置文件目录
            debounce_interval: 防抖间隔（秒）
        """
        logger.info(f"初始化配置监控器: {config_dir}, 防抖间隔: {debounce_interval}秒")
        self._config_dir = config_dir
        self._debounce_interval = debounce_interval
        self._observer = None
        self._debounce_timers = {}
        self._manager = None'''

            content = init_pattern.sub(new_init, content)
            print("__init__方法增强完成")
        else:
            print("未找到__init__方法，跳过增强")

        # 写回文件
        print(f"正在写入文件：{file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("成功写入文件")
        return 0

    except Exception as e:
        print(f"错误：{str(e)}")
        print("详细错误信息：")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
