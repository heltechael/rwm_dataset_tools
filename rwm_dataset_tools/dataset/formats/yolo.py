"""
Base YOLO format handler for dataset creation.
"""
import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, TextIO

from rwm_dataset_tools.dataset.processing import find_relevant_eppo
from rwm_dataset_tools.utils.path import create_directory, create_symlink

logger = logging.getLogger(__name__)

class YOLOFormatBase:
    """
    Base class for YOLO format dataset creation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLO format handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = os.path.expanduser(config['dataset']['output_dir'])
        self.eppo_codes = config['dataset']['eppo_codes']
        
        # Create the directory structure
        self._create_directory_structure()
        
    def _create_directory_structure(self) -> None:
        """
        Create the directory structure for the YOLO dataset.
        """
        # Get directory names from config or use defaults
        structure = self.config['dataset'].get('structure', {})
        self.images_dir = structure.get('images_dir', 'images')
        self.labels_dir = structure.get('labels_dir', 'labels')
        self.train_dir = structure.get('train_dir', 'train')
        self.val_dir = structure.get('val_dir', 'val')
        self.test_dir = structure.get('test_dir', 'test')
        
        # Create main directories
        create_directory(self.output_dir)
        
        # Create images directories
        self.images_path = os.path.join(self.output_dir, self.images_dir)
        self.train_images_path = os.path.join(self.images_path, self.train_dir)
        self.val_images_path = os.path.join(self.images_path, self.val_dir)
        self.test_images_path = os.path.join(self.images_path, self.test_dir)
        
        create_directory(self.images_path)
        create_directory(self.train_images_path)
        create_directory(self.val_images_path)
        create_directory(self.test_images_path)
        
        # Create labels directories
        self.labels_path = os.path.join(self.output_dir, self.labels_dir)
        self.train_labels_path = os.path.join(self.labels_path, self.train_dir)
        self.val_labels_path = os.path.join(self.labels_path, self.val_dir)
        self.test_labels_path = os.path.join(self.labels_path, self.test_dir)
        
        create_directory(self.labels_path)
        create_directory(self.train_labels_path)
        create_directory(self.val_labels_path)
        create_directory(self.test_labels_path)
        
    def get_split_paths(self, split: str) -> Tuple[str, str]:
        """
        Get the paths for a specific dataset split.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (images_path, labels_path) for the specified split
        """
        if split == 'train':
            return self.train_images_path, self.train_labels_path
        elif split == 'val':
            return self.val_images_path, self.val_labels_path
        elif split == 'test':
            return self.test_images_path, self.test_labels_path
        else:
            raise ValueError(f"Invalid split: {split}")
            
    def create_image_symlink(self, source_path: str, image_id: int, split: str) -> str:
        """
        Create a symlink to an image in the dataset.
        
        Args:
            source_path: Path to the source image
            image_id: Image ID
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Path to the created symlink
        """
        # Get the destination directory
        images_dir, _ = self.get_split_paths(split)
        
        # Get the extension from the source path
        _, ext = os.path.splitext(source_path)
        
        # Create the destination path
        dest_path = os.path.join(images_dir, f"{image_id}{ext}")
        
        # Create the symlink
        create_symlink(source_path, dest_path)
        
        return dest_path
    
    def row_to_yolo_format(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Convert a row of annotation data to YOLO format.
        
        Args:
            row: Dictionary with annotation data
            
        Returns:
            YOLO format string or None if the annotation should be skipped
        """
        eppo_code = row['EPPOCode']
        cotyledon_id = row['cotyledon']
        
        # Skip if no EPPO code
        if pd.isna(eppo_code):
            return None
            
        # Find the relevant EPPO code
        eppo_code = find_relevant_eppo(eppo_code, cotyledon_id, self.eppo_codes)
        if eppo_code is None:
            return None
            
        # Get the class index
        class_index = self.eppo_codes.index(eppo_code)
        
        # Get bounding box coordinates
        min_x = row['MinX']
        min_y = row['MinY']
        max_x = row['MaxX']
        max_y = row['MaxY']
        image_width = row['Width']
        image_height = row['Height']
        
        # Calculate box dimensions
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        # Calculate box center
        center_x = float(box_width) / 2 + min_x
        center_y = float(box_height) / 2 + min_y
        
        # Normalize coordinates
        center_x_norm = center_x / float(image_width)
        center_y_norm = center_y / float(image_height)
        box_width_norm = box_width / float(image_width)
        box_height_norm = box_height / float(image_height)
        
        # Create YOLO format string
        return f"{class_index} {center_x_norm} {center_y_norm} {box_width_norm} {box_height_norm}"
    
    def create_label_file(self, annotations: pd.DataFrame, image_id: int, split: str) -> str:
        """
        Create a label file for a single image.
        
        Args:
            annotations: DataFrame with annotations for a single image
            image_id: Image ID
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Path to the created label file
        """
        # Get the destination directory
        _, labels_dir = self.get_split_paths(split)
        
        # Create the destination path
        dest_path = os.path.join(labels_dir, f"{image_id}.txt")
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for _, row in annotations.iterrows():
            yolo_line = self.row_to_yolo_format(row)
            if yolo_line:
                yolo_lines.append(yolo_line)
                
        # Write the label file
        with open(dest_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        return dest_path
        
    def create_dataset_yaml(self) -> str:
        """
        Create the dataset YAML file.
        
        Returns:
            Path to the created YAML file
        """
        yaml_filename = self.config['dataset'].get('yaml_filename', 'dataset.yaml')
        yaml_path = os.path.join(self.output_dir, yaml_filename)
        
        # Create YAML content
        yaml_content = {
            'path': self.output_dir,
            'train': os.path.join(self.images_dir, self.train_dir),
            'val': os.path.join(self.images_dir, self.val_dir),
            'test': os.path.join(self.images_dir, self.test_dir),
            'nc': len(self.eppo_codes),
            'names': self.eppo_codes
        }
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        return yaml_path