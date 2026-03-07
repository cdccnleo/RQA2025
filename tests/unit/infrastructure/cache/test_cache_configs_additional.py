import pytest

from src.infrastructure.cache.core.cache_configs import (
    CacheConfig,
    BasicCacheConfig,
    DistributedCacheConfig,
    CacheLevel,
)


def test_from_dict_strict_invalid_redis_port_raises():
    config_dict = {
        "strict_validation": True,
        "distributed": {"distributed": True, "redis_host": "localhost", "redis_port": 0},
    }

    with pytest.raises(ValueError):
        CacheConfig.from_dict(config_dict)


def test_from_dict_non_strict_recovers_from_bad_basic_and_distributed():
    config_dict = {
        "strict_validation": False,
        "basic": {"max_size": "invalid", "ttl": "bad"},
        "distributed": {"distributed": True, "redis_host": "", "redis_port": 0},
        "multi_level": {"level": CacheLevel.MEMORY.value},
    }

    config = CacheConfig.from_dict(config_dict)

    assert isinstance(config.basic, BasicCacheConfig)
    assert config.basic.max_size == config.max_size
    assert config.distributed.redis_host in ("localhost", "")
    assert isinstance(config.distributed.redis_port, int)
    assert 1 <= config.distributed.redis_port <= 65535


def test_from_dict_syncs_top_level_fields():
    config_dict = {
        "strict_validation": True,
        "max_size": 200,
        "ttl": 50,
        "basic": {"max_size": 100, "ttl": 10},
        "distributed": {"distributed": False},
    }

    config = CacheConfig.from_dict(config_dict)

    assert config.basic.max_size == 200
    assert config.basic.ttl == 50
    assert config.max_size == 200
    assert config.ttl == 50

