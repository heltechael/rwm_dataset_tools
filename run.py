#!/usr/bin/env python3
"""
Main entry point for the RWM dataset tools.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List

from rwm_dataset_tools.utils.config import load_config
from rwm_dataset_tools.dataset.extraction import DatasetExtractor
from rwm_dataset_tools.dataset.formats.yolov5 import YOLOv5Format
from rwm_dataset_tools.dataset.formats.yolov11 import YOLOv11Format

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

file_handler = logging.FileHandler('rwm_extraction.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RWM Dataset Tools')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/models/yolov11.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['yolov5', 'yolov11'], 
        default='yolov11',
        help='Output format'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--debug-db', 
        action='store_true',
        help='Run database structure checks before extraction'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check database and exit without creating dataset files'
    )

    parser.add_argument(
        '--copy-images', 
        action='store_true',
        help='Copy images instead of creating symlinks (useful for faster storage)'
    )

    parser.add_argument(
        '--output-base-dir', 
        type=str, 
        default='/fast_data',
        help='Base directory for output (default: /fast_data)'
    )
    
    return parser.parse_args()

def main() -> None:
    """
    Main entry point
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    logging_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(logging_level)
    
    # Print startup information
    logger.info("=" * 80)
    logger.info("RWM Dataset Extraction Tool - Starting")
    logger.info("=" * 80)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output format: {args.format}")
    if args.output_dir:
        logger.info(f"Output directory (override): {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load configuration
    try:
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error(f"Make sure the config file exists at: {os.path.abspath(args.config)}")
        sys.exit(1)
    
    # Override configuration with command line arguments
    if args.output_dir:
        config['dataset']['output_dir'] = args.output_dir
    else:
        # Set default output directory using the base directory
        dataset_name = os.path.basename(args.config).split('.')[0]
        config['dataset']['output_dir'] = os.path.join(args.output_base_dir, f"rwm_dataset_{args.format}")
        logger.info(f"Using default output directory: {config['dataset']['output_dir']}")

    # Set image copying mode
    config['dataset']['copy_images'] = args.copy_images
    if args.copy_images:
        logger.info("Images will be copied instead of symlinked")
        
    config['random_seed'] = args.seed
    
    # Create database connection for debugging
    if args.debug_db or args.dry_run:
        try:
            from rwm_dataset_tools.database.connection import RWMDatabase
            from rwm_dataset_tools.database.debug import DatabaseDebugger
            
            logger.info("Initializing database connection for debugging...")
            db = RWMDatabase(config['database'])
            debugger = DatabaseDebugger(db)
            
            with db:
                logger.info("Running database structure checks...")
                debugger.check_database_structure()
                
            if args.dry_run:
                logger.info("Dry run completed. Exiting without creating dataset.")
                return
                
        except Exception as e:
            logger.error(f"Database debugging failed: {e}")
            if args.dry_run:
                sys.exit(1)
    
    # Create format handler
    try:
        logger.info(f"Creating {args.format} format handler...")
        if args.format == 'yolov5':
            format_handler = YOLOv5Format(config)
        elif args.format == 'yolov11':
            format_handler = YOLOv11Format(config)
        else:
            raise ValueError(f"Unsupported format: {args.format}")
        logger.info(f"Format handler created. Output dir: {format_handler.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create format handler: {e}")
        sys.exit(1)
    
    # Create dataset extractor
    try:
        logger.info("Creating dataset extractor...")
        extractor = DatasetExtractor(config, format_handler)
        logger.info("Dataset extractor created")
    except Exception as e:
        logger.error(f"Failed to create dataset extractor: {e}")
        sys.exit(1)
    
    # Extract dataset
    try:
        logger.info("Starting dataset extraction process...")
        stats = extractor.extract()
        logger.info("Dataset extraction completed successfully")
    except Exception as e:
        logger.error(f"Dataset extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Log statistics
    logger.info("=" * 80)
    logger.info("DATASET EXTRACTION SUMMARY")
    logger.info("=" * 80)
    if 'error' in stats:
        logger.error(f"Error: {stats['error']}")
    else:
        logger.info(f"Total images processed: {stats['total_images']}")
        logger.info(f"Training set:   {stats['train_images']} images, {stats['train_annotations']} annotations")
        logger.info(f"Validation set: {stats['val_images']} images, {stats['val_annotations']} annotations")
        logger.info(f"Test set:       {stats['test_images']} images, {stats['test_annotations']} annotations")
        logger.info(f"Total annotations: {stats['total_annotations']}")
        
        if 'skipped_images' in stats and stats['skipped_images'] > 0:
            logger.warning(f"Skipped images: {stats['skipped_images']} (source files not found)")
            
        if 'errors' in stats and stats['errors'] > 0:
            logger.warning(f"Errors encountered: {stats['errors']}")
            
        # Calculate some metrics
        if stats['total_images'] > 0:
            logger.info(f"Average annotations per image: {stats['total_annotations'] / stats['total_images']:.2f}")
            
        train_pct = stats['train_images'] / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
        val_pct = stats['val_images'] / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
        test_pct = stats['test_images'] / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
        
        logger.info(f"Split percentages: Train {train_pct:.1f}% / Val {val_pct:.1f}% / Test {test_pct:.1f}%")
        
        # Add output location information
        output_dir = config['dataset']['output_dir']
        logger.info(f"Dataset created in: {os.path.abspath(output_dir)}")
        logger.info(f"Dataset YAML file: {os.path.abspath(os.path.join(output_dir, config['dataset'].get('yaml_filename', 'dataset.yaml')))}")
        
    logger.info("=" * 80)
    logger.info("For detailed logs, check rwm_extraction.log")
    
if __name__ == '__main__':
    main()