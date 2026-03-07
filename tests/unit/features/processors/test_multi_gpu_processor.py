import logging

import pandas as pd
import pytest

from src.features.processors.gpu.multi_gpu_processor import MultiGPUProcessor


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.logger",
        logging.getLogger(__name__),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.get_logger",
        lambda name: logging.getLogger(name),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor",
        lambda config=None: None,
    )


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(
        MultiGPUProcessor,
        "_initialize_multi_gpu",
        lambda self: None,
    )
    monkeypatch.setattr(
        MultiGPUProcessor,
        "_initialize_fallback",
        lambda self: None,
    )
    return MultiGPUProcessor(config={"use_multi_gpu": False, "fallback_to_cpu": True})


@pytest.fixture
def sample_frame():
    return pd.DataFrame({"close": range(12)})


def test_split_data_for_gpus(processor, sample_frame):
    processor.available_gpus = [0, 1, 2]
    processor.config["chunk_size"] = 1
    chunks = processor._split_data_for_gpus(sample_frame)
    assert len(chunks) == 3
    assert sum(len(chunk) for chunk in chunks.values()) == len(sample_frame)


def test_aggregate_results_preserves_index(processor, sample_frame):
    data1 = pd.DataFrame({"feature_a": [1, 2]}, index=sample_frame.index[:2])
    data2 = pd.DataFrame({"feature_a": [3, 4]}, index=sample_frame.index[2:4])
    combined = processor._aggregate_results({0: data1, 1: data2}, sample_frame.index[:4])
    pd.testing.assert_series_equal(
        combined["feature_a"],
        pd.Series([1, 2, 3, 4], index=sample_frame.index[:4], name="feature_a"),
    )


def test_aggregate_results_empty(processor, sample_frame):
    combined = processor._aggregate_results({}, sample_frame.index)
    assert combined.empty


def test_calculate_multiple_indicators_multi_gpu_cpu_fallback(processor, sample_frame, monkeypatch):
    class _SingleGPUStub:
        def calculate_multiple_indicators_gpu(self, data, indicators, params=None):
            return data.assign(foo=1)

    processor.single_gpu_processor = _SingleGPUStub()
    monkeypatch.setattr(
        processor,
        "_calculate_multiple_indicators_cpu",
        lambda data, indicators, params=None: data.assign(foo=1),
    )
    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"], params={})
    assert "foo" in result.columns


def test_calculate_multiple_indicators_multi_gpu_parallel(processor, sample_frame, monkeypatch):
    processor.available_gpus = [0, 1]

    def fake_load_balance(data):
        half = len(data) // 2
        return {
            0: data.iloc[:half],
            1: data.iloc[half:],
        }

    def fake_process_chunk(gpu_id, chunk, indicators, params):
        return chunk.assign(**{f"feature_{gpu_id}": chunk["close"]})

    monkeypatch.setattr(processor, "_load_balance_data", fake_load_balance)
    monkeypatch.setattr(processor, "_process_chunk_on_gpu", fake_process_chunk)

    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"], params={})
    assert {"feature_0", "feature_1"} <= set(result.columns)


def test_initialize_fallback_creates_single_gpu(monkeypatch):
    class DummyGPUProcessor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor",
        DummyGPUProcessor,
    )
    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": True, "fallback_to_cpu": True})
    assert isinstance(proc.single_gpu_processor, DummyGPUProcessor)


def test_initialize_fallback_cpu_only(monkeypatch):
    class BoomGPUProcessor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("gpu init fail")

    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor",
        BoomGPUProcessor,
    )
    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": True, "fallback_to_cpu": True})
    assert getattr(proc, "single_gpu_processor", None) is None


def test_initialize_fallback_without_cpu(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)

    monkeypatch.setattr(MultiGPUProcessor, "single_gpu_processor", None, raising=False)
    proc = MultiGPUProcessor(
        config={
            "fallback_to_single_gpu": False,
            "fallback_to_cpu": False,
        }
    )
    assert getattr(proc, "single_gpu_processor", None) is None
    assert proc.available_gpus == []


def test_initialize_multi_gpu_handles_no_devices(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        True,
    )
    monkeypatch.setattr(MultiGPUProcessor, "_detect_available_gpus", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_create_gpu_processors", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_warmup_gpus", lambda self: None)

    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": True, "fallback_to_cpu": True})
    proc.available_gpus = []
    proc._initialize_multi_gpu()
    assert proc.available_gpus == []


def test_initialize_multi_gpu_failure_triggers_fallback(monkeypatch):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", True)

    def boom_detect(self):
        raise RuntimeError("detect fail")

    called = {}

    def fake_fallback(self):
        called["fallback"] = True

    monkeypatch.setattr(MultiGPUProcessor, "_detect_available_gpus", boom_detect)
    monkeypatch.setattr(MultiGPUProcessor, "_create_gpu_processors", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_warmup_gpus", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", fake_fallback)

    MultiGPUProcessor(config={"fallback_to_single_gpu": False, "fallback_to_cpu": True})
    assert called.get("fallback") is True


def test_split_data_for_gpus_handles_no_gpus(processor, sample_frame):
    processor.available_gpus = []
    chunks = processor._split_data_for_gpus(sample_frame)
    assert chunks == {}


def test_memory_based_distribution_uses_low_memory_gpu(processor, sample_frame, monkeypatch):
    processor.available_gpus = [0, 1]
    processor.config["chunk_size"] = 2

    allocations = {0: 0.1, 1: 0.5}

    class TorchStub:
        class cuda:
            @staticmethod
            def set_device(gpu_id):
                return None

            @staticmethod
            def memory_allocated(gpu_id):
                return allocations[gpu_id] * 1024 ** 3

            @staticmethod
            def memory_reserved(gpu_id):
                return 0.0

    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.torch",
        TorchStub(),
        raising=False,
    )

    chunks = processor._memory_based_distribution(sample_frame)
    assert set(chunks.keys()) == {0, 1}
    # GPU 0 内存使用更少，应先分配较小分片
    assert len(chunks[0]) >= len(chunks[1])


def test_memory_based_distribution_handles_errors(processor, sample_frame, monkeypatch):
    processor.available_gpus = [0]

    class TorchStub:
        class cuda:
            @staticmethod
            def set_device(_):
                return None

            @staticmethod
            def memory_allocated(_):
                raise RuntimeError("alloc fail")

            @staticmethod
            def memory_reserved(_):
                raise RuntimeError("reserve fail")

    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.torch", TorchStub(), raising=False)
    processor.config["chunk_size"] = 2
    chunks = processor._memory_based_distribution(sample_frame)
    assert list(chunks.keys()) == [0]


def test_process_chunk_on_gpu_success(sample_frame, monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    processor = MultiGPUProcessor(config={"use_multi_gpu": False, "fallback_to_cpu": True})
    processor.available_gpus = [0]
    dummy_result = sample_frame.assign(result=1)

    class DummyProcessor:
        def calculate_multiple_indicators_gpu(self, data, indicators, params=None):
            return dummy_result

    processor.gpu_processors = {0: DummyProcessor()}

    class TorchStub:
        class cuda:
            @staticmethod
            def set_device(gpu_id):
                return None

    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.torch",
        TorchStub(),
        raising=False,
    )

    result = processor._process_chunk_on_gpu(0, sample_frame, ["sma"])
    pd.testing.assert_frame_equal(result, dummy_result)


def test_process_chunk_on_gpu_handles_failure(processor, sample_frame):
    processor.available_gpus = [0]
    processor.gpu_processors = {}
    result = processor._process_chunk_on_gpu(0, sample_frame, ["sma"])
    assert result.empty


def test_calculate_multiple_indicators_multi_gpu_async_mode(sample_frame, monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    processor = MultiGPUProcessor(
        config={
            "use_multi_gpu": False,
            "fallback_to_cpu": True,
            "sync_mode": False,
        }
    )
    processor.available_gpus = [0, 1]

    def fake_load(data):
        half = len(data) // 2
        return {0: data.iloc[:half], 1: data.iloc[half:]}

    def fake_process(gpu_id, chunk, indicators, params):
        return chunk.assign(**{f"gpu_{gpu_id}": gpu_id})

    monkeypatch.setattr(processor, "_load_balance_data", fake_load)
    monkeypatch.setattr(processor, "_process_chunk_on_gpu", fake_process)

    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert {"gpu_0", "gpu_1"} <= set(result.columns)


def test_calculate_multiple_indicators_multi_gpu_no_processors_returns_empty(processor, sample_frame):
    processor.available_gpus = [0]
    processor.gpu_processors = {0: None}

    def fake_process(*args, **kwargs):
        return pd.DataFrame()

    processor._process_chunk_on_gpu = fake_process  # type: ignore[assignment]

    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert result.empty


def test_calculate_multiple_indicators_cpu_path(processor, sample_frame, monkeypatch):
    processor.available_gpus = []
    processor.single_gpu_processor = None

    def fake_cpu(data, indicators, params=None):
        return data.assign(cpu_flag=True)

    monkeypatch.setattr(processor, "_calculate_multiple_indicators_cpu", fake_cpu)
    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert "cpu_flag" in result.columns


def test_load_balance_performance_distribution_uses_split(processor, sample_frame, monkeypatch):
    processor.available_gpus = [0, 1]
    called = {}

    def fake_split(data, gpu_list=None):
        called["args"] = gpu_list
        return {0: data.iloc[:1], 1: data.iloc[1:2]}

    processor.config["load_balancing"] = "performance_based"
    monkeypatch.setattr(processor, "_split_data_for_gpus", fake_split)
    chunks = processor._load_balance_data(sample_frame)
    assert set(chunks.keys()) == {0, 1}
    assert called["args"] is None


def test_calculate_multi_gpu_returns_empty_when_no_chunks(processor, sample_frame, monkeypatch, caplog):
    processor.available_gpus = [0]
    monkeypatch.setattr(processor, "_load_balance_data", lambda *_: {})
    with caplog.at_level("WARNING"):
        result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert result.empty


def test_get_available_and_status(processor):
    processor.available_gpus = [0, 1]
    assert processor.get_available_gpus() == [0, 1]
    assert processor.is_multi_gpu_available() is True
    processor.available_gpus = [0]
    assert processor.is_multi_gpu_available() is False


def test_get_multi_gpu_info_handles_error(monkeypatch):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", False)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)
    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": False, "fallback_to_cpu": False})
    proc.available_gpus = [0]
    proc.gpu_processors = {0: None}
    proc.gpu_info = {0: {"name": "gpu0", "memory_gb": 8}}

    class TorchStub:
        class cuda:
            @staticmethod
            def set_device(_):
                raise RuntimeError("fail")

            @staticmethod
            def memory_allocated(_):
                return 0

            @staticmethod
            def memory_reserved(_):
                return 0

    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.torch", TorchStub(), raising=False)
    info = proc.get_multi_gpu_info()
    assert info["available_gpus"] == 1


def test_clear_multi_gpu_memory_calls_processors(monkeypatch):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", False)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    class ProcessorStub:
        def __init__(self):
            self.cleared = False

        def clear_gpu_memory(self):
            self.cleared = True

    stub = ProcessorStub()
    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": False, "fallback_to_cpu": False})
    proc.available_gpus = [0]
    proc.gpu_processors = {0: stub}

    class TorchStub:
        class cuda:
            @staticmethod
            def set_device(_):
                return None

            @staticmethod
            def empty_cache():
                return None

    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.torch", TorchStub(), raising=False)
    proc.clear_multi_gpu_memory()
    assert stub.cleared is True


def test_create_gpu_processors_removes_failed(monkeypatch):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", False)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    class DummyProcessor:
        def __init__(self, config=None):
            self.config = config

    created = {}

    def fake_gpu_processor(config=None):
        device = config.get("device_id")
        if device == 1:
            raise RuntimeError("fail")
        created[device] = config
        return DummyProcessor(config)

    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor",
        fake_gpu_processor,
    )

    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": False, "fallback_to_cpu": False})
    proc.available_gpus = [0, 1]
    proc._create_gpu_processors()
    assert proc.available_gpus == [0]
    assert 0 in proc.gpu_processors
    assert created[0]["device_id"] == 0


def test_warmup_gpus_handles_exception(monkeypatch, sample_frame):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", False)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    class BoomProcessor:
        def calculate_multiple_indicators_gpu(self, *args, **kwargs):
            raise RuntimeError("warmup fail")

    proc = MultiGPUProcessor(config={"fallback_to_single_gpu": False, "fallback_to_cpu": False, "warmup_iterations": 1})
    proc.available_gpus = [0]
    proc.gpu_processors = {0: BoomProcessor()}
    proc._warmup_gpus()  # should not raise


def test_calculate_multiple_indicators_multi_gpu_partial_results(monkeypatch, sample_frame):
    monkeypatch.setattr("src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE", False)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_fallback", lambda self: None)

    processor = MultiGPUProcessor(
        config={
            "use_multi_gpu": False,
            "fallback_to_cpu": True,
            "sync_mode": True,
        }
    )
    processor.available_gpus = [0, 1]

    def fake_load(data):
        half = len(data) // 2
        return {0: data.iloc[:half], 1: data.iloc[half:]}

    def fake_process(gpu_id, chunk, indicators, params):
        if gpu_id == 0:
            return pd.DataFrame()
        return chunk.assign(ok=gpu_id)

    monkeypatch.setattr(processor, "_load_balance_data", fake_load)
    monkeypatch.setattr(processor, "_process_chunk_on_gpu", fake_process)

    result = processor.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert "ok" in result.columns
    assert result["ok"].eq(1).all()


def test_calculate_multiple_indicators_cpu_returns_empty(processor, sample_frame):
    empty = processor._calculate_multiple_indicators_cpu(sample_frame, ["sma"])
    assert empty.empty


def test_calculate_multi_gpu_returns_empty_without_fallback(monkeypatch, sample_frame):
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
        False,
    )
    monkeypatch.setattr(MultiGPUProcessor, "_initialize_multi_gpu", lambda self: None)

    monkeypatch.setattr(MultiGPUProcessor, "single_gpu_processor", None, raising=False)
    proc = MultiGPUProcessor(
        config={
            "fallback_to_single_gpu": False,
            "fallback_to_cpu": False,
        }
    )
    result = proc.calculate_multiple_indicators_multi_gpu(sample_frame, ["sma"])
    assert result.empty

