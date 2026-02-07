"""
Model Trainer Component
Implements AutoML - tests multiple models and selects the best performer.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple, Any
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from src.exception import CustomException
from src.logger import logging
from src.config import model_trainer_config


class ModelTrainer:
    """
    AutoML model trainer that tests multiple algorithms and selects the best.
    """
    
    def __init__(self):
        self.config = model_trainer_config
        logging.info("ModelTrainer component initialized")
    
    def get_model_candidates(self) -> Dict[str, Any]:
        """
        Get dictionary of model candidates to test.
        
        Returns:
            Dict mapping model names to model objects
        """
        models = {}
        
        if self.config.test_naive_bayes:
            models['Multinomial Naive Bayes'] = MultinomialNB(alpha=0.1)
        
        if self.config.test_svm:
            models['Linear SVM'] = LinearSVC(
                C=1.0,
                max_iter=1000,
                random_state=self.config.random_state
            )
        
        if self.config.test_logistic_regression:
            models['Logistic Regression'] = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if self.config.test_random_forest:
            models['Random Forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if self.config.test_xgboost:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        logging.info(f"Testing {len(models)} model candidates: {list(models.keys())}")
        return models
    
    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a single model using cross-validation and test set.
        
        Args:
            model: Model to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dict of evaluation metrics
        """
        try:
            logging.info(f"Evaluating {model_name}...")
            
            # Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'model_name': model_name,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
            
            logging.info(f"{model_name} - F1: {metrics['f1_weighted']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            return {
                'model_name': model_name,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'accuracy': 0.0,
                'precision_weighted': 0.0,
                'recall_weighted': 0.0,
                'f1_weighted': 0.0,
                'error': str(e)
            }
    
    def train_and_select_best(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict]:
        """
        Train all candidate models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (best_model, all_metrics)
        """
        try:
            logging.info("=" * 80)
            logging.info("Starting Model Training and Selection")
            logging.info("=" * 80)
            
            # Get model candidates
            models = self.get_model_candidates()
            
            # Evaluate all models
            all_metrics = []
            trained_models = {}
            
            for model_name, model in models.items():
                metrics = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test, model_name
                )
                all_metrics.append(metrics)
                trained_models[model_name] = model
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(all_metrics)
            comparison_df = comparison_df.sort_values(
                by=self.config.primary_metric,
                ascending=False
            )
            
            logging.info("\nModel Comparison:")
            logging.info(f"\n{comparison_df.to_string()}")
            
            # Select best model
            best_model_name = comparison_df.iloc[0]['model_name']
            best_model = trained_models[best_model_name]
            best_metrics = comparison_df.iloc[0].to_dict()
            
            logging.info(f"\n{'=' * 80}")
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"F1 Score: {best_metrics['f1_weighted']:.4f}")
            logging.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
            logging.info(f"{'=' * 80}")
            
            # Evaluate on validation set
            logging.info("\nEvaluating best model on validation set...")
            y_val_pred = best_model.predict(X_val)
            
            val_metrics = {
                'val_accuracy': float(accuracy_score(y_val, y_val_pred)),
                'val_f1_weighted': float(f1_score(y_val, y_val_pred, average='weighted')),
                'val_precision_weighted': float(precision_score(y_val, y_val_pred, average='weighted', zero_division=0)),
                'val_recall_weighted': float(recall_score(y_val, y_val_pred, average='weighted', zero_division=0)),
            }
            
            logging.info(f"Validation F1: {val_metrics['val_f1_weighted']:.4f}")
            logging.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            # Combine metrics
            final_metrics = {
                **best_metrics,
                **val_metrics,
                'training_date': datetime.now().isoformat(),
                'all_models_comparison': comparison_df.to_dict('records')
            }
            
            # Save comparison
            self.config.model_comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(self.config.model_comparison_path, index=False)
            logging.info(f"Model comparison saved to {self.config.model_comparison_path}")
            
            return best_model, final_metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_model_with_metadata(
        self,
        model: Any,
        metrics: Dict,
        label_encoder: Any
    ) -> None:
        """
        Save model with metadata.
        
        Args:
            model: Trained model
            metrics: Model metrics
            label_encoder: Label encoder used for categories
        """
        try:
            logging.info("Saving model and metadata...")
            
            # Save model
            self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_artifact = {
                'model': model,
                'label_encoder': label_encoder,
                'metrics': metrics,
                'model_name': metrics.get('model_name', 'Unknown'),
                'training_date': metrics.get('training_date', datetime.now().isoformat())
            }
            
            joblib.dump(model_artifact, self.config.model_path)
            logging.info(f"Model saved to {self.config.model_path}")
            
            # Save metrics as JSON
            with open(self.config.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Metrics saved to {self.config.metrics_path}")
            
            logging.info("=" * 80)
            logging.info("Model Training Pipeline Completed Successfully")
            logging.info("=" * 80)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_encoder: Any
    ) -> Tuple[Any, Dict]:
        """
        Main method to execute model training pipeline.
        
        Returns:
            Tuple of (best_model, metrics)
        """
        try:
            # Train and select best model
            best_model, metrics = self.train_and_select_best(
                X_train, y_train, X_test, y_test, X_val, y_val
            )
            
            # Save model with metadata
            self.save_model_with_metadata(best_model, metrics, label_encoder)
            
            return best_model, metrics
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the model trainer component
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # Run full pipeline
    print("Running data ingestion...")
    ingestion = DataIngestion()
    train_path, test_path, val_path = ingestion.initiate_data_ingestion()
    
    print("\nRunning data transformation...")
    transformation = DataTransformation()
    X_train, X_test, X_val, y_train, y_test, y_val = transformation.transform_data(
        train_path, test_path, val_path
    )
    
    print("\nRunning model training...")
    trainer = ModelTrainer()
    best_model, metrics = trainer.initiate_model_training(
        X_train, y_train, X_test, y_test, X_val, y_val,
        transformation.label_encoder
    )
    
    print(f"\nModel training completed!")
    print(f"Best Model: {metrics['model_name']}")
    print(f"F1 Score: {metrics['f1_weighted']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
