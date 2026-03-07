import pandas as pd
import pytest
import sys

import src.data.sources.data_source_manager as dsm


@pytest.fixture(autouse=True)
def _stub_requests(monkeypatch):
    """避免真实 HTTP 调用。"""

    class DummyResponse:
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

    dummy_response = DummyResponse(payload={
        "chart": {
            "result": [
                {
                    "timestamp": [1, 2],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1.0, 2.0],
                                "high": [1.1, 2.2],
                                "low": [0.9, 1.9],
                                "close": [1.05, 2.05],
                                "volume": [100, 200],
                            }
                        ]
                    },
                }
            ]
        }
    })
    class DummyRequests:
        @staticmethod
        def head(*args, **kwargs):
            return dummy_response

        @staticmethod
        def get(*args, **kwargs):
            return dummy_response

    dummy_module = DummyRequests()
    dummy_module.__dict__.update({"head": dummy_module.head, "get": dummy_module.get})
    monkeypatch.setitem(sys.modules, "requests", dummy_module)


def test_yahoo_source_fetch_data_returns_dataframe():
    source = dsm.YahooFinanceSource()
    source.check_availability()
    df = source.fetch_data("TEST", "2025-01-01", "2025-01-02")
    assert not df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_data_source_manager_fetch_data_with_fallback_returns_first_available(monkeypatch):
    manager = dsm.DataSourceManager()

    class FakeSource(dsm.DataSource):
        def check_availability(self):
            self.is_available = True
            return True

        def fetch_data(self, *args, **kwargs):
            return pd.DataFrame({"value": [1]})

    manager.sources["fake"] = FakeSource("fake")

    df = manager.fetch_data_with_fallback("SYM", "2025-01-01", "2025-01-02", preferred_source="fake")
    assert df["value"].iloc[0] == 1

    monkeypatch.setattr(manager.sources["fake"], "check_availability", lambda: False)
    fallback = manager.fetch_data_with_fallback("SYM", "2025-01-01", "2025-01-02", preferred_source="fake")
    assert not fallback.empty

