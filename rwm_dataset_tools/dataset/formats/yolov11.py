"""
YOLOv11 specific format handler for dataset creation.
"""
import os
import logging
from typing import Dict, Any, Optional

from rwm_dataset_tools.dataset.formats.yolo import YOLOFormatBase

logger = logging.getLogger(__name__)

class YOLOv11Format(YOLOFormatBase):
    """
    YOLOv11 format handler for dataset creation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv11 format handler.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # YOLOv11 specific configuration
        self.image_size = config['dataset'].get('image_size', 1280)
        
    def create_dataset_yaml(self) -> str:
        """
        Create the dataset YAML file with YOLOv11 specific options.
        
        Returns:
            Path to the created YAML file
        """
        yaml_path = super().create_dataset_yaml()
        
        # Log YOLOv11 specific configuration
        logger.info(f"Created YOLOv11 dataset YAML with image size {self.image_size}")
        
        return yaml_path