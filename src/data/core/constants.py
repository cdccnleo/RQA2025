"""
数据管理层常量定义
Data Management Layer Constants

定义数据管理相关的常量，避免魔法数字
"""

# 数据加载参数
DEFAULT_BATCH_SIZE = 1000           # 默认批处理大小
MAX_BATCH_SIZE = 10000             # 最大批处理大小
DATA_LOAD_TIMEOUT = 300             # 数据加载超时时间(秒)

# 缓存参数
DATA_CACHE_SIZE = 10000             # 数据缓存大小
CACHE_TTL_SECONDS = 1800            # 缓存过期时间(秒)
CACHE_EVICTION_INTERVAL = 300       # 缓存驱逐间隔(秒)

# 数据质量参数
QUALITY_CHECK_SAMPLE_SIZE = 1000    # 质量检查采样大小
DATA_VALIDITY_THRESHOLD = 0.95      # 数据有效性阈值
DUPLICATE_CHECK_THRESHOLD = 0.01    # 重复检查阈值

# 数据库参数
DB_CONNECTION_TIMEOUT = 30          # 数据库连接超时(秒)
DB_QUERY_TIMEOUT = 60               # 数据库查询超时(秒)
DB_CONNECTION_POOL_SIZE = 20        # 数据库连接池大小

# 文件处理参数
MAX_FILE_SIZE_MB = 500              # 最大文件大小(MB)
FILE_READ_BUFFER_SIZE = 8192        # 文件读取缓冲区大小
COMPRESSION_THRESHOLD_MB = 10       # 压缩阈值(MB)

# 数据转换参数
TRANSFORM_TIMEOUT = 120             # 数据转换超时时间(秒)
MAX_TRANSFORM_RETRIES = 3           # 最大转换重试次数
VALIDATION_SAMPLE_RATIO = 0.1       # 验证采样比例

# 同步参数
SYNC_INTERVAL_SECONDS = 300         # 同步间隔(秒)
SYNC_TIMEOUT_SECONDS = 600          # 同步超时时间(秒)
SYNC_BATCH_SIZE = 5000              # 同步批处理大小

# 监控参数
DATA_MONITOR_INTERVAL = 60          # 数据监控间隔(秒)
QUALITY_ALERT_THRESHOLD = 0.9       # 质量告警阈值
PERFORMANCE_ALERT_THRESHOLD = 0.8   # 性能告警阈值

# 备份参数
BACKUP_RETENTION_DAYS = 30          # 备份保留天数
BACKUP_COMPRESSION_LEVEL = 6        # 备份压缩级别
BACKUP_VERIFICATION_TIMEOUT = 300   # 备份验证超时时间(秒)

# 版本控制参数
VERSION_RETENTION_COUNT = 10        # 版本保留数量
VERSION_CHECK_INTERVAL = 3600       # 版本检查间隔(秒)
ROLLBACK_TIMEOUT = 600              # 回滚超时时间(秒)

# 分布式参数
SHARD_COUNT_DEFAULT = 4             # 默认分片数量
REPLICATION_FACTOR = 3              # 复制因子
CONSISTENCY_LEVEL = "QUORUM"        # 一致性级别

# 预处理参数
PREPROCESSING_TIMEOUT = 180         # 预处理超时时间(秒)
OUTLIER_DETECTION_THRESHOLD = 3.0   # 异常值检测阈值
NORMALIZATION_METHOD = "ZSCORE"     # 标准化方法

# 导出参数
EXPORT_TIMEOUT = 300                # 导出超时时间(秒)
EXPORT_COMPRESSION_ENABLED = True   # 导出压缩启用
EXPORT_FORMAT_DEFAULT = "PARQUET"   # 默认导出格式
