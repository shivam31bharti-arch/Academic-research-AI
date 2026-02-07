"""
Configuration management for Academic Research AI project.
Centralized configuration for all components.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from ArXiv."""
    
    # ArXiv categories to fetch
    categories: List[str] = field(default_factory=lambda: [
        "cs.AI",      # Artificial Intelligence
        "cs.LG",      # Machine Learning
        "cs.CL",      # Computation and Language (NLP)
        "stat.ML",    # Machine Learning (Statistics)
        "cs.CV",      # Computer Vision
    ])
    
    # Number of papers per category
    max_results_per_category: int = 500
    
    # Data paths
    raw_data_path: Path = DATA_DIR / "raw_papers.csv"
    train_data_path: Path = DATA_DIR / "train.csv"
    test_data_path: Path = DATA_DIR / "test.csv"
    val_data_path: Path = DATA_DIR / "val.csv"
    
    # Train-test-val split ratios
    train_size: float = 0.7
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Cache settings
    use_cache: bool = True
    cache_expiry_days: int = 7


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation and preprocessing."""
    
    # Preprocessor path
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl"
    
    # TF-IDF parameters
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    
    # Text preprocessing
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    
    # Feature selection
    use_feature_selection: bool = True
    n_features_to_select: int = 3000


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    
    # Model paths
    model_path: Path = MODELS_DIR / "best_model.pkl"
    metrics_path: Path = MODELS_DIR / "metrics.json"
    model_comparison_path: Path = MODELS_DIR / "model_comparison.csv"
    
    # Cross-validation
    cv_folds: int = 5
    
    # Evaluation metrics
    primary_metric: str = "f1_weighted"  # Used for model selection
    
    # Model candidates to test
    test_naive_bayes: bool = True
    test_svm: bool = True
    test_random_forest: bool = True
    test_xgboost: bool = True
    test_logistic_regression: bool = True
    
    # Hyperparameter tuning
    use_hyperparameter_tuning: bool = True
    n_iter_search: int = 20  # Number of parameter settings sampled
    
    # Random state
    random_state: int = 42


@dataclass
class PredictionConfig:
    """Configuration for prediction pipeline."""
    
    # Confidence threshold
    min_confidence_threshold: float = 0.5
    
    # Batch processing
    batch_size: int = 32
    
    # ArXiv API settings
    arxiv_timeout: int = 30  # seconds


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""
    
    # Monitoring paths
    predictions_log_path: Path = LOGS_DIR / "predictions.csv"
    performance_log_path: Path = LOGS_DIR / "performance.csv"
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.15  # 15% drop in confidence triggers alert
    
    # Retraining triggers
    auto_retrain: bool = False  # Set to True for automatic retraining
    min_predictions_before_retrain: int = 1000
    performance_drop_threshold: float = 0.10  # 10% drop triggers retraining


@dataclass
class WebAppConfig:
    """Configuration for Streamlit web application."""
    
    # App settings
    page_title: str = "Academic Research AI"
    page_icon: str = "📚"
    layout: str = "wide"
    
    # Display settings
    show_confidence_scores: bool = True
    show_model_metrics: bool = True
    max_batch_upload_size: int = 100  # Maximum papers in batch upload


@dataclass
class APIConfig:
    """Configuration for FastAPI."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # API settings
    title: str = "Academic Research AI API"
    description: str = "Automated paper classification using NLP and AutoML"
    version: str = "1.0.0"
    
    # Rate limiting (requests per minute)
    rate_limit: int = 60


# Global configuration instances
data_ingestion_config = DataIngestionConfig()
data_transformation_config = DataTransformationConfig()
model_trainer_config = ModelTrainerConfig()
prediction_config = PredictionConfig()
monitoring_config = MonitoringConfig()
webapp_config = WebAppConfig()
api_config = APIConfig()
