import os
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Created directory: {path}")

def create_symlink(source: str, destination: str, overwrite: bool = False) -> None:
    """
    Create a symbolic link.
    
    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite an existing link
    """
    # Ensure the source exists
    if not os.path.exists(source):
        logger.warning(f"Source path does not exist: {source}")
        raise FileNotFoundError(f"Source file not found: {source}")
        
    # Convert paths to absolute for better logging
    source_abs = os.path.abspath(source)
    dest_abs = os.path.abspath(destination)
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(destination)
    create_directory(parent_dir)
    
    # Remove existing link if overwrite is True
    if os.path.exists(destination):
        if overwrite:
            try:
                os.remove(destination)
                logger.debug(f"Removed existing file: {dest_abs}")
            except Exception as e:
                logger.error(f"Failed to remove existing file {dest_abs}: {e}")
                raise
        else:
            logger.debug(f"Destination already exists: {dest_abs}")
            return
    
    # Create the symbolic link
    try:
        os.symlink(source_abs, dest_abs)
        logger.debug(f"Created symlink: {source_abs} -> {dest_abs}")
    except Exception as e:
        # Check for specific permission errors
        if isinstance(e, PermissionError):
            logger.error(f"Permission denied when creating symlink: {dest_abs}")
            logger.error("Make sure you have write permissions to the destination directory")
        else:
            logger.error(f"Failed to create symlink: {e}")
        raise    

def remove_directory(path: str) -> None:
    """
    Remove a directory and all its contents.
    
    Args:
        path: Directory path
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.debug(f"Removed directory: {path}")

def get_file_extension(path: str) -> str:
    """
    Get the file extension of a path.
    
    Args:
        path: File path
        
    Returns:
        File extension (including the dot)
    """
    return os.path.splitext(path)[1]

    