"""
Configuration utilities for the RWM dataset tools.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Check if there's an inherit directive
    if 'inherit' in config:
        # Get the path to the inherited config
        inherit_path = config['inherit']
        
        # If it's a relative path, resolve it relative to the current config file
        if not os.path.isabs(inherit_path):
            config_dir = os.path.dirname(config_path)
            inherit_path = os.path.join(config_dir, inherit_path)
            
        # Load the inherited config
        inherited_config = load_config(inherit_path)
        
        # Remove the inherit directive
        del config['inherit']
        
        # Merge the configs (current config takes precedence)
        merged_config = merge_configs(inherited_config, config)
        
        return merged_config
    
    return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        # If both values are dictionaries, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        # Otherwise, override the value
        else:
            result[key] = value
            
    return result