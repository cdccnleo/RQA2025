import pytest
from unittest.mock import Mock
from src.data.adapters.base_adapter import BaseDataAdapter, DataModel

class DummyAdapter(BaseDataAdapter):
    @property
    def adapter_type(self) -> str:
        return "dummy"
    
    def load_data(self, config):
        return DataModel(
            raw_data={"test": "data"},
            metadata={"source": "dummy"},
            validation_status=True
        )
    
    def validate(self, data: DataModel) -> bool:
        return data.validation_status

def test_base_adapter_init():
    adapter = DummyAdapter()
    assert isinstance(adapter, BaseDataAdapter)
    assert adapter.adapter_type == "dummy"

def test_base_adapter_load_data():
    adapter = DummyAdapter()
    config = {"test": "config"}
    result = adapter.load_data(config)
    assert isinstance(result, DataModel)
    assert result.raw_data == {"test": "data"}
    assert result.metadata == {"source": "dummy"}
    assert result.validation_status is True

def test_base_adapter_validate():
    adapter = DummyAdapter()
    data = DataModel(
        raw_data={"test": "data"},
        metadata={"source": "dummy"},
        validation_status=True
    )
    assert adapter.validate(data) is True
    
    invalid_data = DataModel(
        raw_data={"test": "data"},
        metadata={"source": "dummy"},
        validation_status=False
    )
    assert adapter.validate(invalid_data) is False

def test_base_adapter_str_representation():
    adapter = DummyAdapter()
    assert "DummyAdapter" in str(adapter)
    assert "dummy" in str(adapter)

def test_data_model_creation():
    data_model = DataModel(
        raw_data={"key": "value"},
        metadata={"source": "test"},
        validation_status=True
    )
    assert data_model.raw_data == {"key": "value"}
    assert data_model.metadata == {"source": "test"}
    assert data_model.validation_status is True 