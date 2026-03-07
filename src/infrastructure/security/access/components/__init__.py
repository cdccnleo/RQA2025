from importlib import import_module

access_checker = import_module(".access_checker", __name__)

__all__ = ["access_checker"]

