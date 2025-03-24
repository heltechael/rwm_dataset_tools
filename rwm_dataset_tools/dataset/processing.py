"""
Processing functions for the RWM dataset.
"""
import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def partition_by_image_id(data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Partition annotation data by image ID.
    
    Args:
        data: DataFrame with annotation data
        
    Returns:
        Dictionary mapping image IDs to DataFrames with annotations
    """
    return {img_id: group for img_id, group in data.groupby('ImageId')}

def process_psez_annotations(data: pd.DataFrame, psez_crops: List[str]) -> pd.DataFrame:
    """
    Process PSEZ annotations, filtering those that are not inside a crop box.
    
    Args:
        data: DataFrame with annotation data
        psez_crops: List of EPPO codes for crops that PSEZ should be associated with
        
    Returns:
        DataFrame with processed annotations
    """
    logger.info("Processing PSEZ annotations...")
    
    # Separate PSEZ and non-PSEZ annotations
    psez_data = data[data['EPPOCode'] == 'PSEZ'].copy()
    non_psez_data = data[data['EPPOCode'] != 'PSEZ'].copy()
    
    logger.info(f"Found {len(psez_data)} PSEZ annotations")
    
    # Initialize list to store PSEZ annotations that are enclosed by a crop
    valid_psez_indices = []
    
    # For each PSEZ annotation
    for idx, psez_row in psez_data.iterrows():
        image_id = psez_row['ImageId']
        
        # Check if there's a crop box in the same image that encloses this PSEZ
        crop_boxes = non_psez_data[
            (non_psez_data['ImageId'] == image_id) & 
            (non_psez_data['EPPOCode'].isin(psez_crops))
        ]
        
        for _, crop_row in crop_boxes.iterrows():
            # Check if the center of PSEZ is inside the crop box
            if center_enclosed(
                inner_box=np.array([
                    psez_row['MinX'], psez_row['MinY'], 
                    psez_row['MaxX'], psez_row['MaxY']
                ]),
                outer_box=np.array([
                    crop_row['MinX'], crop_row['MinY'], 
                    crop_row['MaxX'], crop_row['MaxY']
                ])
            ):
                # This PSEZ is inside a crop box
                valid_psez_indices.append(idx)
                logger.debug(f"PSEZ {psez_row['Id']} is enclosed by crop {crop_row['Id']} ({crop_row['EPPOCode']})")
                break
    
    # Get all valid PSEZ annotations
    valid_psez_data = psez_data.loc[valid_psez_indices]
    
    logger.info(f"Found {len(valid_psez_data)} valid PSEZ annotations inside crop boxes")
    logger.info(f"Filtered out {len(psez_data) - len(valid_psez_data)} PSEZ annotations")
    
    # Combine valid PSEZ and non-PSEZ annotations
    result = pd.concat([non_psez_data, valid_psez_data])
    
    return result

def center_enclosed(inner_box: np.ndarray, outer_box: np.ndarray) -> bool:
    """
    Check if the center of inner_box is enclosed by outer_box.
    
    Args:
        inner_box: numpy array [min_x, min_y, max_x, max_y] for inner box
        outer_box: numpy array [min_x, min_y, max_x, max_y] for outer box
        
    Returns:
        True if center of inner_box is inside outer_box, False otherwise
    """
    # Calculate center of inner box
    width = inner_box[2] - inner_box[0]
    center_x = inner_box[0] + width/2.0
    
    height = inner_box[3] - inner_box[1]
    center_y = inner_box[1] + height/2.0
    
    # Check if center is inside outer box
    return (
        center_x > outer_box[0] and  # x1
        center_y > outer_box[1] and  # y1
        center_x < outer_box[2] and  # x2
        center_y < outer_box[3]      # y2
    )

def find_relevant_eppo(eppo_code: str, cotyledon_id: int, eppo_codes: List[str]) -> Optional[str]:
    """
    Find the relevant EPPO code for an annotation, following the same logic as I-GIS scripts.
    
    Args:
        eppo_code: Original EPPO code
        cotyledon_id: Cotyledon ID
        eppo_codes: List of valid EPPO codes
        
    Returns:
        Relevant EPPO code or None if no relevant code found
    """
    # Handle prefixes (e.g., SOLTU1 -> SOLTU)
    for valid_eppo in eppo_codes:
        if eppo_code.startswith(valid_eppo):
            eppo_code = valid_eppo
    
    # Find the EPPO code to use
    if eppo_code in eppo_codes:
        return eppo_code
    elif cotyledon_id == -100:
        return 'PPPMM'  # Monocot
    elif cotyledon_id == -101:
        return 'PPPDD'  # Dicot
    else:
        return None

def determine_dataset_split(
    data_rows: pd.DataFrame, 
    config: Dict[str, Any],
    rng: np.random.RandomState = None
) -> str:
    """
    Determine which dataset split (train/val/test) an image belongs to.
    
    Args:
        data_rows: DataFrame with annotation data for a single image
        config: Configuration dictionary
        rng: Random number generator for reproducibility
        
    Returns:
        String indicating the split: 'train', 'val', or 'test'
    """
    if rng is None:
        rng = np.random.RandomState()
        
    # Get the first row (they all have the same image ID)
    row = data_rows.iloc[0]
    upload_id = row['UploadId']
    image_id = row['ImageId']
    grown_weed = row['GrownWeed']
    
    # Check fixed upload lists
    fixed_sets = config['dataset']['fixed_sets']
    if upload_id in fixed_sets['train_uploads']:
        return 'train'
    elif upload_id in fixed_sets['val_uploads']:
        return 'val'
    elif upload_id in fixed_sets['test_uploads']:
        return 'test'
    
    # Check fixed image lists
    if image_id in fixed_sets['train_images']:
        return 'train'
    elif image_id in fixed_sets['val_images']:
        return 'val'
    elif image_id in fixed_sets['test_images']:
        return 'test'
    
    # Fix grown images to train
    if grown_weed:
        return 'train'
    
    # Distribute at random based on probabilities
    split_probs = config['dataset']['split_probabilities']
    probabilities = np.array([
        split_probs['train'],
        split_probs['val'],
        split_probs['test']
    ])
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum()
    
    # Choose a split
    split_idx = rng.choice(3, p=probabilities)
    return ['train', 'val', 'test'][split_idx]

def parse_poly_data(poly_data_str: str) -> List[Dict[str, Any]]:
    """
    Parse the PolyData JSON string to a list of dictionaries.
    
    Args:
        poly_data_str: PolyData JSON string
        
    Returns:
        List of dictionaries with polygon data
    """
    if pd.isna(poly_data_str):
        return []
    
    try:
        return json.loads(poly_data_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse PolyData: {poly_data_str}")
        return []