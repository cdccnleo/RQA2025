import pytest
from infrastructure import config


@pytest.fixture
def fpga_mode():
    """标记当前测试模式"""
    return pytest.mark.skipif(
        config.FPGA_ENABLED,
        reason="当前运行在软件降级模式"
    )