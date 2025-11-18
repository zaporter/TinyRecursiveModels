import importlib
import inspect
from typing import Any


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def load_callable(identifier: str, prefix: str = "") -> Any:
    """
    Load an arbitrary callable defined as `module_path@symbol`.
    """
    module_path, symbol_name = identifier.split('@')
    module_name = f"{prefix}{module_path}" if prefix else module_path
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)
