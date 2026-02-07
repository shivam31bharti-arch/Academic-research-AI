"""
Prediction Pipeline
Handles predictions on new academic papers.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from pathlib import Path
import joblib
import arxiv

from src.exception import CustomException
from src.logger import logging
from src.config import prediction_config, model_trainer_config, data_transformation_config


class PredictionPipeline:
    """
    Pipeline for making predictions on new academic papers.
    """
    
    def __init__(self):
        self.config = prediction_config
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.load_artifacts()
        logging.info("Prediction Pipeline initialized")
    
    def load_artifacts(self):
        """Load trained model and preprocessor."""
        try:
            # Load model
            model_path = model_trainer_config.model_path
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Please train the model first using: "
                    "python -m src.pipeline.train_pipeline"
                )
            
            model_artifact = joblib.load(model_path)
            self.model = model_artifact['model']
            self.label_encoder = model_artifact['label_encoder']
            logging.info(f"Model loaded: {model_artifact['model_name']}")
            
            # Load preprocessor
            preprocessor_path = data_transformation_config.preprocessor_path
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
            preprocessor_artifact = joblib.load(preprocessor_path)
            self.preprocessor = preprocessor_artifact['preprocessor']
            logging.info("Preprocessor loaded successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Make prediction on a single text.
        
        Args:
            text: Input text (title + abstract or just abstract)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dict with prediction results
        """
        try:
            # Preprocess
            text = self.preprocess_text(text)
            
            if not text:
                return {
                    'error': 'Empty text provided',
                    'predicted_category': None,
                    'confidence': 0.0
                }
            
            # Transform
            X = self.preprocessor.transform([text])
            
            # Predict
            prediction = self.model.predict(X)[0]
            predicted_category = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get probabilities if model supports it
            confidence = 0.0
            probabilities = {}
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = float(proba.max())
                
                if return_probabilities:
                    for idx, prob in enumerate(proba):
                        category = self.label_encoder.inverse_transform([idx])[0]
                        probabilities[category] = float(prob)
            elif hasattr(self.model, 'decision_function'):
                # For SVM
                decision = self.model.decision_function(X)[0]
                confidence = float(np.max(decision))
            
            result = {
                'predicted_category': predicted_category,
                'confidence': confidence,
                'probabilities': probabilities if return_probabilities else None
            }
            
            logging.info(f"Prediction: {predicted_category} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise CustomException(e, sys)
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        try:
            logging.info(f"Making batch predictions for {len(texts)} texts...")
            results = []
            
            for text in texts:
                result = self.predict_single(text, return_probabilities=False)
                results.append(result)
            
            logging.info(f"Batch prediction completed: {len(results)} results")
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_arxiv_id(self, arxiv_id: str) -> Dict:
        """
        Fetch paper from ArXiv and make prediction.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            Dict with paper info and prediction
        """
        try:
            logging.info(f"Fetching paper from ArXiv: {arxiv_id}")
            
            # Clean arxiv_id
            arxiv_id = arxiv_id.strip().replace('arxiv:', '').replace('arXiv:', '')
            
            # Fetch from ArXiv
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results(), None)
            
            if paper is None:
                return {
                    'error': f'Paper not found: {arxiv_id}',
                    'arxiv_id': arxiv_id
                }
            
            # Combine title and abstract
            text = f"{paper.title} {paper.summary}"
            
            # Make prediction
            prediction = self.predict_single(text)
            
            # Add paper metadata
            result = {
                'arxiv_id': arxiv_id,
                'title': paper.title,
                'abstract': paper.summary[:500] + '...' if len(paper.summary) > 500 else paper.summary,
                'published': paper.published.isoformat() if paper.published else None,
                'authors': [author.name for author in paper.authors][:3],  # First 3 authors
                **prediction
            }
            
            logging.info(f"ArXiv prediction completed for {arxiv_id}")
            return result
            
        except Exception as e:
            logging.error(f"ArXiv prediction error: {e}")
            return {
                'error': str(e),
                'arxiv_id': arxiv_id
            }
    
    def predict_from_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Make predictions on a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            
        Returns:
            DataFrame with predictions added
        """
        try:
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
            logging.info(f"Making predictions on DataFrame with {len(df)} rows...")
            
            predictions = self.predict_batch(df[text_column].tolist())
            
            df['predicted_category'] = [p['predicted_category'] for p in predictions]
            df['confidence'] = [p['confidence'] for p in predictions]
            
            logging.info("DataFrame predictions completed")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the prediction pipeline
    print("\n" + "=" * 80)
    print("ACADEMIC RESEARCH AI - PREDICTION PIPELINE TEST")
    print("=" * 80 + "\n")
    
    pipeline = PredictionPipeline()
    
    # Test 1: Single text prediction
    print("Test 1: Single Text Prediction")
    print("-" * 80)
    test_text = """
    Deep Learning for Natural Language Processing
    This paper presents a comprehensive survey of deep learning techniques 
    applied to natural language processing tasks including text classification,
    machine translation, and question answering.
    """
    result = pipeline.predict_single(test_text)
    print(f"Text: {test_text[:100]}...")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    if result['probabilities']:
        print("\nTop 3 Probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in sorted_probs:
            print(f"  {cat}: {prob:.4f}")
    
    # Test 2: ArXiv ID prediction
    print("\n\nTest 2: ArXiv ID Prediction")
    print("-" * 80)
    print("Enter an ArXiv ID (e.g., 2301.12345) or press Enter to skip:")
    arxiv_id = input("> ").strip()
    
    if arxiv_id:
        result = pipeline.predict_from_arxiv_id(arxiv_id)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Title: {result['title']}")
            print(f"Predicted Category: {result['predicted_category']}")
            print(f"Confidence: {result['confidence']:.4f}")
    
    print("\n" + "=" * 80)
    print("Prediction pipeline test completed!")
    print("=" * 80 + "\n")
