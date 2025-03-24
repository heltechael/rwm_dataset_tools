"""
Database debugging utilities.
"""
import logging
import pandas as pd
from typing import Dict, Any, List, Optional

from rwm_dataset_tools.database.connection import RWMDatabase

logger = logging.getLogger(__name__)

class DatabaseDebugger:
    """
    Utility class for debugging database issues.
    """
    def __init__(self, db: RWMDatabase):
        """
        Initialize the debugger.
        
        Args:
            db: Database connection
        """
        self.db = db
        
    def check_database_structure(self) -> None:
        """
        Check the database structure and log information about key tables.
        """
        logger.info("Checking database structure...")
        
        try:
            # Get list of tables
            query = """
            SELECT 
                t.TABLE_SCHEMA,
                t.TABLE_NAME,
                (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS c 
                 WHERE c.TABLE_SCHEMA = t.TABLE_SCHEMA AND c.TABLE_NAME = t.TABLE_NAME) AS COLUMN_COUNT
            FROM 
                INFORMATION_SCHEMA.TABLES t
            WHERE 
                t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY 
                t.TABLE_SCHEMA, t.TABLE_NAME
            """
            tables = self.db.execute_query(query)
            
            logger.info(f"Found {len(tables)} tables in the database")
            
            # Log some key tables
            key_tables = tables[tables['TABLE_NAME'].isin(['Images', 'Annotations', 'AnnotationData', 'PlantInfo', 'Uploads'])]
            logger.info("Key tables:")
            for _, row in key_tables.iterrows():
                logger.info(f"  {row['TABLE_SCHEMA']}.{row['TABLE_NAME']} ({row['COLUMN_COUNT']} columns)")
                
            # Check key tables structure
            self._check_table_structure('Images')
            self._check_table_structure('Annotations')
            self._check_table_structure('AnnotationData')
            self._check_table_structure('PlantInfo')
            
            # Check table counts
            self._check_table_count('Images')
            self._check_table_count('Annotations')
            self._check_table_count('AnnotationData')
            self._check_table_count('PlantInfo')
            
            # Check for UseForTraining flag
            self._check_training_flags()
            
            # Check for "empty" annotations (missing key fields)
            self._check_annotation_data_quality()
            
        except Exception as e:
            logger.error(f"Error checking database structure: {e}")
            
    def _check_table_structure(self, table_name: str, schema: str = 'data') -> None:
        """
        Check the structure of a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: 'data')
        """
        query = f"""
        SELECT 
            COLUMN_NAME, 
            DATA_TYPE, 
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE
        FROM 
            INFORMATION_SCHEMA.COLUMNS
        WHERE 
            TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
        ORDER BY 
            ORDINAL_POSITION
        """
        
        try:
            columns = self.db.execute_query(query)
            logger.info(f"Table {schema}.{table_name} structure ({len(columns)} columns):")
            
            # Only log a subset of columns for readability
            important_cols = ['Id', 'ImageId', 'UploadId', 'AnnotationId', 'PlantId', 'EPPOCode', 
                             'UseForTraining', 'IsDeleted', 'FileName', 'PolyData', 'MinX', 'MinY', 'MaxX', 'MaxY']
            
            for col in important_cols:
                col_info = columns[columns['COLUMN_NAME'] == col]
                if not col_info.empty:
                    col_row = col_info.iloc[0]
                    logger.info(f"  {col_row['COLUMN_NAME']}: {col_row['DATA_TYPE']} (Nullable: {col_row['IS_NULLABLE']})")
                    
        except Exception as e:
            logger.error(f"Error checking table structure for {schema}.{table_name}: {e}")
            
    def _check_table_count(self, table_name: str, schema: str = 'data') -> None:
        """
        Check the row count of a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: 'data')
        """
        query = f"SELECT COUNT(*) AS count FROM [{schema}].[{table_name}]"
        
        try:
            result = self.db.execute_query(query)
            count = result.iloc[0]['count']
            logger.info(f"Table {schema}.{table_name} contains {count} rows")
            
            # For Images table, check how many are not deleted
            if table_name == 'Images':
                query = f"SELECT COUNT(*) AS count FROM [{schema}].[{table_name}] WHERE IsDeleted = 0"
                result = self.db.execute_query(query)
                active_count = result.iloc[0]['count']
                logger.info(f"  Active images (IsDeleted = 0): {active_count}")
                
            # For Annotations table, check UseForTraining
            if table_name == 'Annotations':
                query = f"SELECT COUNT(*) AS count FROM [{schema}].[{table_name}] WHERE UseForTraining = 1"
                result = self.db.execute_query(query)
                training_count = result.iloc[0]['count']
                logger.info(f"  Training annotations (UseForTraining = 1): {training_count}")
                
        except Exception as e:
            logger.error(f"Error checking row count for {schema}.{table_name}: {e}")
            
    def _check_training_flags(self) -> None:
        """
        Check the UseForTraining flag distribution.
        """
        try:
            query = """
            SELECT 
                a.UseForTraining,
                COUNT(*) AS count
            FROM 
                [data].[Annotations] a
            GROUP BY 
                a.UseForTraining
            """
            result = self.db.execute_query(query)
            
            logger.info("UseForTraining flag distribution:")
            for _, row in result.iterrows():
                flag_value = row['UseForTraining']
                count = row['count']
                logger.info(f"  UseForTraining = {flag_value}: {count} annotations")
                
        except Exception as e:
            logger.error(f"Error checking UseForTraining flags: {e}")
            
    def _check_annotation_data_quality(self) -> None:
        """
        Check for "empty" annotations (missing key fields).
        """
        try:
            query = """
            SELECT 
                COUNT(*) AS count
            FROM 
                [data].[AnnotationData]
            WHERE 
                (MinX IS NULL OR MinY IS NULL OR MaxX IS NULL OR MaxY IS NULL) 
                AND AnnotationId IN (SELECT ImageId FROM [data].[Annotations] WHERE UseForTraining = 1)
            """
            result = self.db.execute_query(query)
            empty_count = result.iloc[0]['count']
            
            logger.info(f"Found {empty_count} annotations marked for training with missing bounding box coordinates")
            
            # Check for NULL EPPOCodes
            query = """
            SELECT 
                COUNT(*) AS count
            FROM 
                [data].[AnnotationData] ad
                LEFT JOIN [data].[PlantInfo] pi ON ad.PlantId = pi.Id
            WHERE 
                pi.EPPOCode IS NULL
                AND ad.AnnotationId IN (SELECT ImageId FROM [data].[Annotations] WHERE UseForTraining = 1)
            """
            result = self.db.execute_query(query)
            null_eppo_count = result.iloc[0]['count']
            
            logger.info(f"Found {null_eppo_count} annotations marked for training with NULL EPPO codes")
            
        except Exception as e:
            logger.error(f"Error checking annotation data quality: {e}")