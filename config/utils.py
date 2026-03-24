import os
import importlib.util
import logging
import json
import yaml
from typing import Dict, Any
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_path(type: str, path: str):
    if type == "model":
        return os.path.join(os.environ["MYGO_MODEL_PATH"], path)
    elif type == "data":
        return os.path.join(os.environ["MYGO_DATASET_PATH"], path)
    elif type == "prompt":
        return os.path.join(os.environ["MYGO_PROJECT_PATH"], "prompts", path)
    else:
        raise ValueError(f"Invalid type: {type}")


class ConfigItem:
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize ConfigItem with a dictionary.
        
        Args:
            data: Dictionary to wrap for dot-notation access
        """
        self._data = data
    
    def __getattr__(self, name: str):
        """
        Allow dot-notation access to dictionary keys.
        
        Args:
            name: Attribute name to access
            
        Returns:
            Value from the dictionary, wrapped in ConfigItem if it's a dict
        """
        if name.startswith("_"):
            return super().__getattribute__(name)
        
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigItem(value)
            else:
                return value
        else:
            raise AttributeError(f"ConfigItem has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """
        Allow setting attributes through dot notation.
        
        Args:
            name: Attribute name to set
            value: Value to set
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, "_data"):
                super().__setattr__(name, value)
            else:
                self._data[name] = value
    
    def __getitem__(self, key: str):
        """
        Allow dictionary-style access.
        
        Args:
            key: Key to access
            
        Returns:
            Value from the dictionary, wrapped in ConfigItem if it's a dict
        """
        value = self._data[key]
        if isinstance(value, dict):
            return ConfigItem(value)
        else:
            return value
    
    def __setitem__(self, key: str, value: Any):
        """
        Allow dictionary-style assignment.
        
        Args:
            key: Key to set
            value: Value to set
        """
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in the data.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._data
    
    def keys(self):
        """Return dictionary keys."""
        for key in self._data.keys():
            if isinstance(key, dict):
                yield ConfigItem(key)
            else:
                yield key
    
    def values(self):
        """Return dictionary values."""
        for value in self._data.values():
            if isinstance(value, dict):
                yield ConfigItem(value)
            else:
                yield value
    
    def items(self):
        """Return dictionary items."""
        for key, value in self._data.items():
            if isinstance(value, dict):
                yield key, ConfigItem(value)
            else:
                yield key, value
    
    def __iter__(self):
        """
        Allow iteration over keys, enabling ** unpacking.
        
        Returns:
            Iterator over dictionary keys
        """
        return iter(self._data)
    
    def update(self, other: Dict[str, Any]):
        """
        Update the ConfigItem with key-value pairs from another dictionary.
        
        Args:
            other: Dictionary to update from
        """
        if not isinstance(other, dict):
            raise TypeError("update() argument must be a dictionary")
        self._data.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ConfigItem back to a regular dictionary.
        
        Returns:
            Deep copy of the internal data dictionary
        """
        return copy.deepcopy(self._data)
    
    def save_json(self, path: str):
        """
        Save the configuration data to a JSON file.
        
        Args:
            path: File path to save to
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._data, f, indent=4, ensure_ascii=False, default=str
            )
        logger.info(f"ConfigItem saved to JSON file: {path}")
    
    def save_yaml(self, path: str):
        """
        Save the configuration data to a YAML file.
        
        Args:
            path: File path to save to
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=4,
            )
        logger.info(f"ConfigItem saved to YAML file: {path}")
    
    def __repr__(self) -> str:
        """Return string representation of ConfigItem."""
        return f"ConfigItem({self._data})"
    
    def __str__(self) -> str:
        """Return string representation of ConfigItem."""
        return str(self._data)


def load_config(config_path: str) -> ConfigItem:
    """
    Load configuration from a Python file and return a ConfigItem.
    
    Args:
        config_path: Path to the configuration Python file
        
    Returns:
        ConfigItem instance containing all dictionary variables from the config file
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ImportError: If there's an error importing the config module
    """
    logger.info(f"Loading config from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config_data = {}
    for attr_name in dir(config_module):
        if not attr_name.startswith("_"):
            attr_value = getattr(config_module, attr_name)
            if isinstance(attr_value, dict):
                config_data[attr_name] = attr_value
    
    logger.info(f"Loaded {len(config_data)} configuration sections")
    return ConfigItem(config_data)