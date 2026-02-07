"""
Training Pipeline
Orchestrates the entire training workflow from data ingestion to model training.
"""

import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    End-to-end training pipeline that orchestrates all components.
    """
    
    def __init__(self):
        logging.info("Training Pipeline initialized")
    
    def run(self):
        """
        Execute the complete training pipeline.
        
        Returns:
            Dict with training results and metrics
        """
        try:
            logging.info("\n" + "=" * 80)
            logging.info("STARTING AUTOMATED TRAINING PIPELINE")
            logging.info("=" * 80 + "\n")
            
            # Step 1: Data Ingestion
            logging.info("STEP 1: Data Ingestion")
            logging.info("-" * 80)
            data_ingestion = DataIngestion()
            train_path, test_path, val_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("\nSTEP 2: Data Transformation")
            logging.info("-" * 80)
            data_transformation = DataTransformation()
            X_train, X_test, X_val, y_train, y_test, y_val = data_transformation.transform_data(
                train_path, test_path, val_path
            )
            
            # Step 3: Model Training
            logging.info("\nSTEP 3: Model Training and Selection")
            logging.info("-" * 80)
            model_trainer = ModelTrainer()
            best_model, metrics = model_trainer.initiate_model_training(
                X_train, y_train, X_test, y_test, X_val, y_val,
                data_transformation.label_encoder
            )
            
            # Summary
            logging.info("\n" + "=" * 80)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            logging.info(f"\nBest Model: {metrics['model_name']}")
            logging.info(f"Test F1 Score: {metrics['f1_weighted']:.4f}")
            logging.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"Validation F1 Score: {metrics['val_f1_weighted']:.4f}")
            logging.info(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
            logging.info("=" * 80 + "\n")
            
            return {
                'status': 'success',
                'model_name': metrics['model_name'],
                'test_f1': metrics['f1_weighted'],
                'test_accuracy': metrics['accuracy'],
                'val_f1': metrics['val_f1_weighted'],
                'val_accuracy': metrics['val_accuracy'],
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ACADEMIC RESEARCH AI - AUTOMATED TRAINING PIPELINE")
    print("=" * 80 + "\n")
    
    print("This will:")
    print("1. Fetch papers from ArXiv API")
    print("2. Preprocess and transform data")
    print("3. Train and evaluate multiple models")
    print("4. Select and save the best model")
    print("\nThis may take 5-15 minutes depending on your internet speed and CPU.")
    print("\nStarting pipeline...\n")
    
    pipeline = TrainingPipeline()
    result = pipeline.run()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nBest Model: {result['model_name']}")
    print(f"Test Accuracy: {result['test_accuracy']:.2%}")
    print(f"Test F1 Score: {result['test_f1']:.4f}")
    print(f"Validation Accuracy: {result['val_accuracy']:.2%}")
    print(f"Validation F1 Score: {result['val_f1']:.4f}")
    print("\nYou can now run the web app with: streamlit run app.py")
    print("=" * 80 + "\n")
