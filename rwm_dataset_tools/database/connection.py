"""
Database connection handling for RWM dataset extraction.
"""
import pyodbc
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")
logger = logging.getLogger(__name__)

class RWMDatabase:
    """
    Class for handling database connections to the RoboWeedMaps database.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database connection with configuration.
        
        Args:
            config: Database configuration dictionary containing server, name, user, password, driver
        """
        self.config = config
        self.connection = None
        self.conn_str = (
            f"DRIVER={{{config['driver']}}};"
            f"SERVER={config['server']};"
            f"DATABASE={config['name']};"
            f"UID={config['user']};"
            f"PWD={config['password']};"
        )
        
    def connect(self) -> None:
        """
        Establish a connection to the RWM database.
        """
        logger.info(f"Connecting to database {self.config['name']} on {self.config['server']}...")
        logger.debug(f"Connection string: {self.conn_str.replace(self.config['password'], '********')}")
        
        try:
            self.connection = pyodbc.connect(self.conn_str)
            logger.info(f"Connected to database {self.config['name']} on {self.config['server']}")
            
            # Test the connection with a simple query
            cursor = self.connection.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            logger.info(f"SQL Server version: {version.split()[0]}")
            
            # Get basic database info
            cursor.execute("SELECT COUNT(*) FROM [data].[Images]")
            image_count = cursor.fetchone()[0]
            logger.info(f"Total images in database: {image_count}")
            
            cursor.execute("SELECT COUNT(*) FROM [data].[Annotations]")
            annotation_count = cursor.fetchone()[0]
            logger.info(f"Total annotations in database: {annotation_count}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Please check:")
            logger.error("1. Database server is running and accessible")
            logger.error("2. Database credentials are correct")
            logger.error("3. ODBC driver is installed")
            logger.error("4. Network connectivity to the database server")
            raise
            
    def disconnect(self) -> None:
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
            
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query string
            params: Optional parameters for the query
            
        Returns:
            DataFrame containing query results
        """
        if not self.connection:
            self.connect()
            
        try:
            if params:
                return pd.read_sql(query, self.connection, params=params)
            else:
                return pd.read_sql(query, self.connection)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Params: {params}")
            raise
            
    def get_table_count(self, table_name: str, schema: str = "data") -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: "data")
            
        Returns:
            Row count as an integer
        """
        query = f"SELECT COUNT(*) AS count FROM [{schema}].[{table_name}]"
        result = self.execute_query(query)
        return result.iloc[0]['count']
        
    def __enter__(self):
        """
        Context manager entry point.
        """
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        """
        self.disconnect()