"""
机器学习层常量定义
ML Layer Constants

定义所有魔法数字和常量，避免硬编码
"""

# 时间相关常量
CACHE_TTL_SECONDS = 3600  # 缓存过期时间1小时
DEFAULT_TIMEOUT = 300     # 默认超时时间5分钟

# 数据处理常量
DEFAULT_TEST_SIZE = 0.2   # 默认测试集比例
DEFAULT_RANDOM_STATE = 42  # 默认随机种子
DEFAULT_CV_FOLDS = 5      # 默认交叉验证折数

# 模型训练常量
DEFAULT_BATCH_SIZE = 32   # 默认批次大小
DEFAULT_EPOCHS = 100      # 默认训练轮数
MAX_EPOCHS = 1000         # 最大训练轮数
MIN_EPOCHS = 1            # 最小训练轮数

# 性能阈值常量
MIN_ACCURACY = 0.5        # 最低准确率阈值
MAX_TRAIN_TIME = 3600     # 最大训练时间(秒)
MEMORY_THRESHOLD_MB = 1024  # 内存阈值(MB)

# 特征处理常量
MAX_FEATURES = 1000       # 最大特征数
MIN_SAMPLES_SPLIT = 2     # 决策树最小样本分割
MAX_DEPTH_DEFAULT = 10    # 默认最大深度

# 超参数搜索常量
GRID_SEARCH_CV = 3        # 网格搜索交叉验证折数
RANDOM_SEARCH_ITER = 50   # 随机搜索迭代次数

# 分布式训练常量
DEFAULT_NUM_WORKERS = 4   # 默认工作进程数
MAX_WORKERS = 16          # 最大工作进程数

# 监控常量
METRICS_UPDATE_INTERVAL = 60  # 指标更新间隔(秒)
ALERT_THRESHOLD = 0.8         # 告警阈值

# 文件路径常量
MODEL_SAVE_SUFFIX = '.pkl'
CONFIG_SAVE_SUFFIX = '.json'

# 状态码常量
SUCCESS_CODE = 0
ERROR_CODE = -1
WARNING_CODE = 1
