"""
优化层常量定义
Optimization Layer Constants

定义优化相关的常量，避免魔法数字
"""

# 优化算法参数
DEFAULT_POPULATION_SIZE = 100     # 默认种群大小
DEFAULT_MAX_GENERATIONS = 200     # 默认最大代数
DEFAULT_MUTATION_RATE = 0.1       # 默认变异率
DEFAULT_CROSSOVER_RATE = 0.8      # 默认交叉率

# 性能阈值
TARGET_FITNESS_THRESHOLD = 0.95   # 目标适应度阈值
MIN_IMPROVEMENT_THRESHOLD = 0.01  # 最小改进阈值
CONVERGENCE_THRESHOLD = 0.001     # 收敛阈值

# 时间限制
MAX_OPTIMIZATION_TIME = 3600      # 最大优化时间(秒)
DEFAULT_TIMEOUT = 300             # 默认超时时间(秒)

# 资源限制
MAX_MEMORY_USAGE_MB = 2048        # 最大内存使用(MB)
MAX_CPU_USAGE_PERCENT = 80        # 最大CPU使用率(%)

# 组合优化参数
DEFAULT_RISK_FREE_RATE = 0.02     # 默认无风险利率
DEFAULT_TARGET_RETURN = 0.08      # 默认目标收益率
DEFAULT_MAX_WEIGHT = 1.0          # 默认最大权重
DEFAULT_MIN_WEIGHT = 0.0          # 默认最小权重

# 超参数搜索
GRID_SEARCH_CV_FOLDS = 5          # 网格搜索交叉验证折数
RANDOM_SEARCH_ITERATIONS = 100    # 随机搜索迭代次数
BAYESIAN_OPT_ITERATIONS = 50      # 贝叶斯优化迭代次数

# 并行处理
DEFAULT_NUM_WORKERS = 4           # 默认工作进程数
MAX_WORKERS = 16                  # 最大工作进程数

# 缓存设置
CACHE_SIZE_LIMIT = 1000           # 缓存大小限制
CACHE_TTL_SECONDS = 1800          # 缓存过期时间(秒)

# 评估指标
MIN_SHARPE_RATIO = 0.5            # 最小夏普比率
MAX_DRAWDOWN_LIMIT = 0.2          # 最大回撤限制
MIN_CALMAR_RATIO = 0.5            # 最小卡玛比率

# 系统优化
BUFFER_SIZE_DEFAULT = 1024        # 默认缓冲区大小
QUEUE_SIZE_DEFAULT = 1000         # 默认队列大小
BATCH_SIZE_DEFAULT = 64           # 默认批处理大小

# 学习率和步长
DEFAULT_LEARNING_RATE = 0.001     # 默认学习率
MIN_LEARNING_RATE = 1e-6          # 最小学习率
MAX_LEARNING_RATE = 1.0           # 最大学习率

# 容差和精度
DEFAULT_TOLERANCE = 1e-6          # 默认容差
HIGH_PRECISION_TOLERANCE = 1e-9   # 高精度容差

# 权重约束
DEFAULT_REBALANCE_THRESHOLD = 0.05  # 默认再平衡阈值
DEFAULT_TRANSACTION_COST = 0.001   # 默认交易成本
