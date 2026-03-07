import pytest

from src.infrastructure.utils.adapters import database_adapter


class FakeTime:
    def __init__(self, start: float):
        self._current = start

    def time(self) -> float:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += seconds


def test_database_adapter_disconnect_clears_connection():
    adapter = database_adapter.DatabaseAdapter()
    adapter.connection = object()

    assert adapter.disconnect() is True
    assert adapter.connection is None


def test_mock_connection_execute_inserts_and_updates():
    conn = database_adapter.MockDatabaseConnection(initial_data={"existing": ("row",)})

    cursor = conn.execute("INSERT INTO table VALUES (%s)", params=("id-1", "value"))
    assert "id-1" in cursor.data

    cursor = conn.execute("UPDATE table SET value=%s WHERE id=%s", params=("updated", "id-1"))
    assert "updated" in cursor.data

    assert cursor.fetchone() is not None
    assert isinstance(cursor.fetchall(), list)


def test_connection_pool_reuses_connections():
    pool = database_adapter.DatabaseConnectionPool(max_size=2, min_size=1)

    with pool.get_connection() as conn:
        conn_id = conn.connection_id
        assert conn_id in pool._in_use  # noqa: SLF001

    # connection should return to available list for reuse
    available_ids = {c.connection_id for c in pool._available}  # noqa: SLF001
    assert conn_id in available_ids


def test_connection_pool_exhaustion_raises():
    pool = database_adapter.DatabaseConnectionPool(max_size=1, min_size=1)

    first = pool._acquire_connection()  # noqa: SLF001
    try:
        with pytest.raises(RuntimeError):
            pool._acquire_connection()  # noqa: SLF001
    finally:
        pool._release_connection(first)  # noqa: SLF001


def test_connection_pool_health_check_reports_leaks(monkeypatch):
    fake_time = FakeTime(1_000.0)
    monkeypatch.setattr(database_adapter.time, "time", fake_time.time)

    pool = database_adapter.DatabaseConnectionPool(max_size=2, min_size=1, leak_detection=True)
    conn = pool._acquire_connection()  # noqa: SLF001

    # simulate leak by advancing time beyond threshold
    fake_time.advance(400.0)
    stats = pool.health_check()
    assert stats["leaks"], "Expected leak to be reported after long hold"

    pool._release_connection(conn)  # noqa: SLF001
    pool.close_all()


def test_connection_pool_close_all_clears_state():
    pool = database_adapter.DatabaseConnectionPool(max_size=3, min_size=2)
    pool.close_all()

    assert not pool._connections  # noqa: SLF001
    assert not pool._available  # noqa: SLF001
    assert not pool._in_use  # noqa: SLF001

