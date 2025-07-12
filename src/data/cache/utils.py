def format_cache_key(key):
    """格式化缓存键"""
    return f"cache_{key}"

def check_expiry(timestamp):
    """检查缓存是否过期"""
    from datetime import datetime
    return datetime.now() > timestamp

def serialize_data(data):
    """序列化缓存数据"""
    import pickle
    return pickle.dumps(data)

def deserialize_data(data):
    """反序列化缓存数据"""
    import pickle
    return pickle.loads(data)
