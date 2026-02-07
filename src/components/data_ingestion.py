"""
Data Ingestion Component
Automatically fetches academic papers from ArXiv API and prepares dataset.
"""

import os
import sys
import pandas as pd
import arxiv
from typing import Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging
from src.config import data_ingestion_config


class DataIngestion:
    """
    Handles automatic data acquisition from ArXiv API.
    Downloads papers, validates data, and splits into train/test/val sets.
    """
    
    def __init__(self):
        self.config = data_ingestion_config
        logging.info("DataIngestion component initialized")
    
    def fetch_from_arxiv(self) -> pd.DataFrame:
        """
        Fetch papers from ArXiv API for specified categories.
        
        Returns:
            pd.DataFrame: DataFrame with columns [title, abstract, category]
        """
        try:
            logging.info("Starting ArXiv data fetching...")
            all_papers = []
            
            for category in self.config.categories:
                logging.info(f"Fetching papers for category: {category}")
                
                # Create search query
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=self.config.max_results_per_category,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                # Fetch papers with progress bar
                papers_fetched = 0
                for result in tqdm(search.results(), 
                                 desc=f"Fetching {category}", 
                                 total=self.config.max_results_per_category):
                    
                    paper_data = {
                        'title': result.title,
                        'abstract': result.summary,
                        'category': category,
                        'published': result.published,
                        'arxiv_id': result.entry_id.split('/')[-1]
                    }
                    all_papers.append(paper_data)
                    papers_fetched += 1
                    
                    if papers_fetched >= self.config.max_results_per_category:
                        break
                
                logging.info(f"Fetched {papers_fetched} papers for {category}")
            
            # Create DataFrame
            df = pd.DataFrame(all_papers)
            logging.info(f"Total papers fetched: {len(df)}")
            logging.info(f"Categories distribution:\n{df['category'].value_counts()}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the fetched data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid
        """
        try:
            logging.info("Validating fetched data...")
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            # Check required columns
            required_columns = ['title', 'abstract', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                logging.warning(f"Null values found:\n{null_counts}")
                # Remove rows with null values
                df.dropna(subset=required_columns, inplace=True)
                logging.info(f"Removed rows with null values. Remaining: {len(df)}")
            
            # Check minimum samples per category
            category_counts = df['category'].value_counts()
            min_samples = 50
            if (category_counts < min_samples).any():
                logging.warning(f"Some categories have less than {min_samples} samples")
            
            logging.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, test, and validation sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        try:
            logging.info("Splitting data into train/test/val sets...")
            
            # Shuffle data
            df = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
            
            # Calculate split indices
            n = len(df)
            train_end = int(n * self.config.train_size)
            test_end = train_end + int(n * self.config.test_size)
            
            # Split data
            train_df = df[:train_end]
            test_df = df[train_end:test_end]
            val_df = df[test_end:]
            
            logging.info(f"Train set: {len(train_df)} samples")
            logging.info(f"Test set: {len(test_df)} samples")
            logging.info(f"Validation set: {len(val_df)} samples")
            
            # Log category distribution
            logging.info(f"Train categories:\n{train_df['category'].value_counts()}")
            
            return train_df, test_df, val_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_data(self, df: pd.DataFrame, filepath: Path) -> None:
        """Save DataFrame to CSV file."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            logging.info(f"Data saved to {filepath}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_cached_data(self) -> pd.DataFrame:
        """Load cached raw data if available and not expired."""
        try:
            if not self.config.use_cache:
                return None
            
            if not self.config.raw_data_path.exists():
                return None
            
            # Check if cache is expired
            file_modified = datetime.fromtimestamp(
                self.config.raw_data_path.stat().st_mtime
            )
            cache_age = datetime.now() - file_modified
            
            if cache_age.days > self.config.cache_expiry_days:
                logging.info(f"Cache expired (age: {cache_age.days} days)")
                return None
            
            logging.info(f"Loading cached data from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded {len(df)} papers from cache")
            return df
            
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None
    
    def initiate_data_ingestion(self) -> Tuple[str, str, str]:
        """
        Main method to execute data ingestion pipeline.
        
        Returns:
            Tuple of (train_path, test_path, val_path)
        """
        try:
            logging.info("=" * 80)
            logging.info("Starting Data Ingestion Pipeline")
            logging.info("=" * 80)
            
            # Try to load cached data
            df = self.load_cached_data()
            
            # If no cache, fetch from ArXiv
            if df is None:
                logging.info("No valid cache found. Fetching from ArXiv...")
                df = self.fetch_from_arxiv()
                
                # Save raw data for caching
                self.save_data(df, self.config.raw_data_path)
            
            # Validate data
            self.validate_data(df)
            
            # Split data
            train_df, test_df, val_df = self.split_data(df)
            
            # Save splits
            self.save_data(train_df, self.config.train_data_path)
            self.save_data(test_df, self.config.test_data_path)
            self.save_data(val_df, self.config.val_data_path)
            
            logging.info("=" * 80)
            logging.info("Data Ingestion Pipeline Completed Successfully")
            logging.info("=" * 80)
            
            return (
                str(self.config.train_data_path),
                str(self.config.test_data_path),
                str(self.config.val_data_path)
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the data ingestion component
    obj = DataIngestion()
    train_path, test_path, val_path = obj.initiate_data_ingestion()
    print(f"\nData ingestion completed!")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Validation data: {val_path}")
