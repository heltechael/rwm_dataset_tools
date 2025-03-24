"""
SQL queries and data fetching for the RWM database.
"""
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from rwm_dataset_tools.database.connection import RWMDatabase

logger = logging.getLogger(__name__)

class RWMDataExtractor:
    """
    Class to extract annotation and image data from the RWM database.
    """
    def __init__(self, db: RWMDatabase, config: Dict[str, Any]):
        """
        Initialize the data extractor.
        
        Args:
            db: Database connection
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        
    def get_annotation_data(self) -> pd.DataFrame:
        """
        Get annotation data for training, following the same logic as the I-GIS scripts.
        This query replicates the logic in the get_labled_data_annotation method.
        
        Returns:
            DataFrame with annotation data
        """
        # Define blacklist plant IDs - typically these would come from configuration
        blacklist_plant_ids = [-12, -7, 0, 148, 150, 151, 994]  # Same as in blacklist_plant_ids_annotation.csv
        
        # Format the blacklist for SQL
        blacklist_str = ', '.join(str(id) for id in blacklist_plant_ids)
        
        # Build the query - this replicates the logic in rwm_db.get_labled_data_annotation()
        query = f"""
        SELECT
            [data].[AnnotationData].[Id],
            [UploadId],
            [FileName],
            [ImageId],
            [PlantId],
            TRIM([data].[PlantInfo].[EPPOCode]) AS EPPOCode,
            [NameEnglish],
            [GrowthStage],
            [Width],
            [Height],
            [PolyData],
            [BrushSize],
            [MinX],
            [MinY],
            [MaxX],
            [MaxY],
            [AnnotationModelId],
            [UseForTraining],
            [ClassificationModelId],
            [Approved],
            [GrownWeed],
            [cotyledon]
         FROM
            [data].[Images]
            INNER JOIN [data].[Annotations] ON ([data].[Annotations].[ImageId] = [data].[Images].[Id])
            LEFT JOIN [data].[AnnotationData] ON ([data].[AnnotationData].[AnnotationId] = [data].[Images].[Id])
            LEFT JOIN [data].[PlantInfo] ON ([AnnotationData].[PlantId] = [data].[PlantInfo].[Id])
            LEFT JOIN [data].[Uploads]  ON ([data].[Images].[UploadId] = [data].[Uploads].[Id])
         WHERE
            [data].[Images].[IsDeleted] = 0
            AND [data].[Uploads].[IsDeleted] = 0
            AND ([data].[AnnotationData].IsTemporary = 0 OR [data].[AnnotationData].IsTemporary is NULL)
            AND [data].[Annotations].[UseForTraining] = 1
            AND [data].[AnnotationData].[PlantId] NOT IN ({blacklist_str})
        """
        
        # Log the actual query for debugging
        logger.debug("SQL Query for fetching annotations:")
        logger.debug(query)
        
        # First check that at least some data exists
        try:
            count_query = """
            SELECT COUNT(*) AS count FROM [data].[Images]
            INNER JOIN [data].[Annotations] ON ([data].[Annotations].[ImageId] = [data].[Images].[Id])
            WHERE [data].[Images].[IsDeleted] = 0
            AND [data].[Annotations].[UseForTraining] = 1
            """
            count_result = self.db.execute_query(count_query)
            img_count = count_result.iloc[0]['count']
            logger.info(f"Found {img_count} images with UseForTraining=1 in the database")
            
            if img_count == 0:
                logger.error("No images with UseForTraining=1 found in database!")
                logger.error("Cannot extract dataset without training images.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error checking for training images: {e}")
            # Continue and try the main query anyway
        
        logger.info("Fetching annotation data from database...")
        try:
            start_time = time.time()
            data = self.db.execute_query(query)
            elapsed = time.time() - start_time
            
            if len(data) == 0:
                logger.error("Query returned zero annotations! Check database content.")
                # Try to get more information about why the query returned no data
                logger.error("Checking AnnotationData table...")
                self.db.execute_query("SELECT TOP 10 Id, AnnotationId FROM [data].[AnnotationData]")
                logger.error("Checking Annotations table...")
                self.db.execute_query("SELECT TOP 10 ImageId, UseForTraining FROM [data].[Annotations]")
            else:
                logger.info(f"Fetched {len(data)} annotation records in {elapsed:.2f} seconds")
                logger.info(f"Data includes {data['ImageId'].nunique()} unique images and {data['EPPOCode'].nunique()} unique EPPO codes")
                
                # Log most common EPPO codes for verification
                logger.info("Most common EPPO codes in the dataset:")
                eppo_counts = data['EPPOCode'].value_counts().head(10)
                for eppo, count in eppo_counts.items():
                    logger.info(f"  {eppo}: {count} annotations")
            
            return data
            
        except Exception as e:
            logger.error(f"Error executing annotation query: {e}")
            logger.error("Make sure the database schema matches what the query expects")
            raise
        
    def filter_held_back_images(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out held back images from the dataset.
        
        Args:
            data: DataFrame with annotation data
            
        Returns:
            Filtered DataFrame
        """
        held_back_images = self.config['dataset']['held_back_images']
        before_len = len(data)
        data = data[~data['ImageId'].isin(held_back_images)]
        logger.info(f"Filtered out held back images: {before_len} -> {len(data)} annotations")
        return data
        
    def get_upload_ids_for_image_ids(self, image_ids: List[int]) -> Dict[int, int]:
        """
        Get upload IDs for a list of image IDs.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            Dictionary mapping image IDs to upload IDs
        """
        # Convert list to string for IN clause
        image_ids_str = ', '.join(str(id) for id in image_ids)
        
        query = f"""
        SELECT [Id] AS ImageId, [UploadId]
        FROM [data].[Images]
        WHERE [Id] IN ({image_ids_str})
        """
        
        result = self.db.execute_query(query)
        return dict(zip(result['ImageId'], result['UploadId']))
        
    def get_image_path(self, upload_id: int, filename: str) -> str:
        """
        Get the full path to an image.
        
        Args:
            upload_id: Upload ID
            filename: Filename
            
        Returns:
            Full path to the image
        """
        rwm_data_path = self.config['paths']['rwm_data']
        return f"{rwm_data_path}/{upload_id}/{filename}"