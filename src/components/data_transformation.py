"""
Data Transformation Component
Handles NLP preprocessing and feature engineering.
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import string
from typing import Tuple
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.exception import CustomException
from src.logger import logging
from src.config import data_transformation_config


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DataTransformation:
    """
    Handles text preprocessing and feature extraction for academic papers.
    """
    
    def __init__(self):
        self.config = data_transformation_config
        self.label_encoder = LabelEncoder()
        self.stop_words = set(stopwords.words('english'))
        logging.info("DataTransformation component initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        try:
            if pd.isna(text):
                return ""
            
            # Convert to string and lowercase
            text = str(text)
            if self.config.lowercase:
                text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove numbers if configured
            if self.config.remove_numbers:
                text = re.sub(r'\d+', '', text)
            
            # Remove punctuation if configured
            if self.config.remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove stopwords if configured
            if self.config.remove_stopwords:
                tokens = word_tokenize(text)
                tokens = [word for word in tokens if word not in self.stop_words]
                text = ' '.join(tokens)
            
            return text
            
        except Exception as e:
            logging.warning(f"Error cleaning text: {e}")
            return ""
    
    def get_preprocessing_pipeline(self) -> Pipeline:
        """
        Create sklearn pipeline for text preprocessing and feature extraction.
        
        Returns:
            Pipeline object
        """
        try:
            logging.info("Creating preprocessing pipeline...")
            
            steps = []
            
            # TF-IDF Vectorization
            tfidf = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                sublinear_tf=True,  # Use sublinear tf scaling
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english'
            )
            steps.append(('tfidf', tfidf))
            
            # Feature selection (optional)
            if self.config.use_feature_selection:
                feature_selector = SelectKBest(
                    chi2,
                    k=min(self.config.n_features_to_select, self.config.max_features)
                )
                steps.append(('feature_selection', feature_selector))
            
            pipeline = Pipeline(steps)
            logging.info(f"Pipeline created with {len(steps)} steps")
            
            return pipeline
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform_data(
        self,
        train_path: str,
        test_path: str,
        val_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform train, test, and validation data.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            val_path: Path to validation data CSV
            
        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        try:
            logging.info("=" * 80)
            logging.info("Starting Data Transformation Pipeline")
            logging.info("=" * 80)
            
            # Load data
            logging.info("Loading data files...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)
            
            logging.info(f"Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")
            
            # Combine title and abstract for better features
            logging.info("Combining title and abstract...")
            train_df['text'] = train_df['title'] + ' ' + train_df['abstract']
            test_df['text'] = test_df['title'] + ' ' + test_df['abstract']
            val_df['text'] = val_df['title'] + ' ' + val_df['abstract']
            
            # Clean text
            logging.info("Cleaning text data...")
            train_df['text'] = train_df['text'].apply(self.clean_text)
            test_df['text'] = test_df['text'].apply(self.clean_text)
            val_df['text'] = val_df['text'].apply(self.clean_text)
            
            # Encode labels
            logging.info("Encoding category labels...")
            y_train = self.label_encoder.fit_transform(train_df['category'])
            y_test = self.label_encoder.transform(test_df['category'])
            y_val = self.label_encoder.transform(val_df['category'])
            
            logging.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            logging.info(f"Classes: {self.label_encoder.classes_}")
            
            # Create and fit preprocessing pipeline
            logging.info("Creating and fitting preprocessing pipeline...")
            preprocessor = self.get_preprocessing_pipeline()
            
            # Fit on training data and transform all sets
            logging.info("Transforming text to features...")
            X_train = preprocessor.fit_transform(train_df['text'], y_train)
            X_test = preprocessor.transform(test_df['text'])
            X_val = preprocessor.transform(val_df['text'])
            
            logging.info(f"Feature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
            
            # Save preprocessor and label encoder
            logging.info("Saving preprocessor and label encoder...")
            self.config.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            
            artifacts = {
                'preprocessor': preprocessor,
                'label_encoder': self.label_encoder,
                'feature_names': preprocessor.named_steps['tfidf'].get_feature_names_out() if hasattr(preprocessor.named_steps['tfidf'], 'get_feature_names_out') else None
            }
            
            joblib.dump(artifacts, self.config.preprocessor_path)
            logging.info(f"Preprocessor saved to {self.config.preprocessor_path}")
            
            logging.info("=" * 80)
            logging.info("Data Transformation Pipeline Completed Successfully")
            logging.info("=" * 80)
            
            return X_train, X_test, X_val, y_train, y_test, y_val
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_preprocessor(self):
        """Load saved preprocessor and label encoder."""
        try:
            if not self.config.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {self.config.preprocessor_path}")
            
            artifacts = joblib.load(self.config.preprocessor_path)
            self.label_encoder = artifacts['label_encoder']
            
            logging.info("Preprocessor loaded successfully")
            return artifacts['preprocessor']
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the data transformation component
    from src.components.data_ingestion import DataIngestion
    
    # First run data ingestion
    ingestion = DataIngestion()
    train_path, test_path, val_path = ingestion.initiate_data_ingestion()
    
    # Then run transformation
    transformation = DataTransformation()
    X_train, X_test, X_val, y_train, y_test, y_val = transformation.transform_data(
        train_path, test_path, val_path
    )
    
    print(f"\nData transformation completed!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_val shape: {X_val.shape}")
