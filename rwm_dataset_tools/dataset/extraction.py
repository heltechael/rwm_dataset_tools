"""
Dataset extraction from RWM database to YOLO format.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from rwm_dataset_tools.database.connection import RWMDatabase
from rwm_dataset_tools.database.queries import RWMDataExtractor
from rwm_dataset_tools.dataset.processing import partition_by_image_id, process_psez_annotations, determine_dataset_split

logger = logging.getLogger(__name__)

class DatasetExtractor:
    """
    Extract dataset from RWM database and prepare it in the required format.
    """
    def __init__(self, config: Dict[str, Any], format_handler):
        """
        Initialize the dataset extractor.
        
        Args:
            config: Configuration dictionary
            format_handler: Format handler instance (e.g., YOLOv11Format)
        """
        self.config = config
        self.format_handler = format_handler
        self.db = RWMDatabase(config['database'])
        self.data_extractor = RWMDataExtractor(self.db, config)
        
        # For reproducibility
        self.random_seed = config.get('random_seed', 42)
        self.rng = np.random.RandomState(self.random_seed)
        
    def extract(self) -> Dict[str, int]:
        """
        Extract dataset from RWM database and prepare it in the required format.
        
        Returns:
            Dictionary with statistics about the extracted dataset
        """
        logger.info("Starting dataset extraction")
        
        # Connect to the database
        with self.db:
            try:
                # Get annotation data
                logger.info("Fetching annotation data from database...")
                start_time = time.time()
                data = self.data_extractor.get_annotation_data()
                elapsed = time.time() - start_time
                logger.info(f"Fetched {len(data)} annotations in {elapsed:.2f} seconds")
                
                if len(data) == 0:
                    logger.error("No annotations found! Check SQL query and database content.")
                    logger.error("Ensure 'UseForTraining' flag is set for images in the database.")
                    return {"error": "No annotations found"}
                
                # Show sample of data
                logger.info("Sample annotation data:")
                sample_data = data.head(3)
                for i, row in sample_data.iterrows():
                    logger.info(f"  Image: {row['ImageId']}, Upload: {row['UploadId']}, EPPO: {row['EPPOCode']}")
                
                # Filter out held back images
                logger.info("Filtering out held back images...")
                data = self.data_extractor.filter_held_back_images(data)
                
                # Process PSEZ annotations
                logger.info("Processing PSEZ annotations...")
                data = process_psez_annotations(data, self.config['dataset']['psez_crops'])
                
                # Partition data by image ID
                logger.info("Partitioning data by image ID...")
                data_by_image = partition_by_image_id(data)
                logger.info(f"Dataset contains {len(data_by_image)} unique images")
                
                # Create dataset files
                logger.info("Creating dataset files...")
                stats = self._create_dataset_files(data_by_image)
                
                # Create dataset YAML file
                logger.info("Creating dataset YAML file...")
                yaml_path = self.format_handler.create_dataset_yaml()
                logger.info(f"Created dataset YAML file: {yaml_path}")
                
                return stats
                
            except Exception as e:
                logger.error(f"Error during dataset extraction: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
    def _create_dataset_files(self, data_by_image: Dict[int, pd.DataFrame]) -> Dict[str, int]:
        """
        Create dataset files (images and labels) for each image.
        
        Args:
            data_by_image: Dictionary mapping image IDs to DataFrames with annotations
            
        Returns:
            Dictionary with statistics about the created files
        """
        # Initialize statistics
        stats = {
            'total_images': len(data_by_image),
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'total_annotations': 0,
            'train_annotations': 0,
            'val_annotations': 0,
            'test_annotations': 0,
            'skipped_images': 0,
            'errors': 0
        }
        
        # Process each image
        logger.info(f"Creating dataset files for {len(data_by_image)} images")
        
        # Use tqdm with a format that shows more information
        progress_bar = tqdm(
            data_by_image.items(), 
            desc="Processing images",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for image_id, annotations in progress_bar:
            try:
                # Update progress description with current image
                progress_bar.set_description(f"Processing image {image_id}")
                
                # Determine dataset split
                split = determine_dataset_split(annotations, self.config, self.rng)
                
                # Get image path
                row = annotations.iloc[0]
                upload_id = row['UploadId']
                filename = row['FileName']
                
                # Get full image path
                source_path = self.data_extractor.get_image_path(upload_id, filename)
                
                # Check if source image exists
                if not os.path.exists(source_path):
                    logger.warning(f"Image not found: {source_path} (UploadId: {upload_id}, ImageId: {image_id})")
                    stats['skipped_images'] += 1
                    continue
                
                # Create image symlink
                try:
                    self.format_handler.create_image_file(source_path, image_id, split)
                except Exception as e:
                    logger.warning(f"Failed to create file for image {image_id}: {e}")
                    stats['errors'] += 1
                    continue
                
                # Create label file
                try:
                    self.format_handler.create_label_file(annotations, image_id, split)
                except Exception as e:
                    logger.warning(f"Failed to create label file for image {image_id}: {e}")
                    stats['errors'] += 1
                    continue
                
                # Update statistics
                stats[f'{split}_images'] += 1
                stats[f'{split}_annotations'] += len(annotations)
                stats['total_annotations'] += len(annotations)
                
                # Log occasional progress details
                if image_id % 100 == 0:
                    logger.debug(f"Processed image {image_id} ({len(annotations)} annotations, split: {split})")
                    
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                stats['errors'] += 1
        
        # Log summary of skipped images
        if stats['skipped_images'] > 0:
            logger.warning(f"Skipped {stats['skipped_images']} images due to missing source files")
        
        # Log summary of errors
        if stats['errors'] > 0:
            logger.warning(f"Encountered {stats['errors']} errors during dataset creation")
        
        return stats