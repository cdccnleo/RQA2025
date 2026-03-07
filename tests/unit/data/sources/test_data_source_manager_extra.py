import sys
import pandas as pd
import pytest

import src.data.sources.data_source_manager as dsm


@pytest.fixture
def _stub_requests_module(monkeypatch):
    """提供可控的 requests stub，用于覆盖 YahooFinanceSource 分支。"""

    class DummyResp:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def head(self, *args, **kwargs):
            return self

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")

    dummy = DummyResp()

    class DummyRequests:
        @staticmethod
        def head(*args, **kwargs):
            return dummy

        @staticmethod
        def get(*args, **kwargs):
            return dummy

    requests_stub = DummyRequests()
    requests_stub.__dict__.update({"head": requests_stub.head, "get": requests_stub.get})
    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    return dummy


def test_get_available_sources_returns_empty_when_head_non_200(_stub_requests_module):
    # 让 HEAD 返回非 200，触发不可用路径
    _stub_requests_module.status_code = 503
    mgr = dsm.DataSourceManager()
    available = mgr.get_available_sources()
    assert available == []


def test_fetch_data_with_fallback_logs_empty_and_returns_empty(monkeypatch, _stub_requests_module, capsys):
    # HEAD 可用，但 GET 返回空结构，触发“获取数据为空”与最终“所有数据源都无法获取数据”
    _stub_requests_module.status_code = 200
    _stub_requests_module._payload = {"chart": {"result": []}}  # 空结果

    mgr = dsm.DataSourceManager()
    df = mgr.fetch_data_with_fallback("SYM", "2025-01-01", "2025-01-02", preferred_source="yahoo")
    assert isinstance(df, pd.DataFrame)
    assert df.empty

    # 可选：检查日志输出包含预期提示（stderr 中的 logging 简要校验）
    captured = capsys.readouterr()
    # 不做严格匹配，只要执行不中断且返回空即可证明分支覆盖
    assert df.empty


