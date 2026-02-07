# 📚 Academic Research AI

> Automated Academic Research Paper Classification System using NLP and AutoML

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, self-sufficient ML system that automatically downloads academic papers from ArXiv, trains classification models, and deploys predictions with **minimal human intervention**.

## ✨ Features

- 🤖 **Fully Automated Data Pipeline** - Auto-downloads papers from ArXiv API
- 🧠 **AutoML Model Selection** - Tests 5+ algorithms and selects the best
- 🔄 **Self-Updating** - Monitors performance and can retrain when needed
- 🌐 **Web Interface** - Easy-to-use Streamlit app for predictions
- 📊 **Real-time Monitoring** - Tracks model performance automatically
- 🚀 **One-Command Deployment** - Single command to train and deploy

## 🎯 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/shivam31bharti-arch/Academic-research-AI.git
cd Academic-research-AI

# Install dependencies
pip install -r requirements.txt
```

### Train the Model (One Command!)

```bash
python -m src.pipeline.train_pipeline
```

This will:
1. ✅ Fetch 2,500 papers from ArXiv API (500 per category)
2. ✅ Preprocess and transform data using NLP techniques
3. ✅ Train and evaluate 5 different models
4. ✅ Select and save the best model
5. ✅ Generate performance metrics

**Expected time:** 5-15 minutes (depending on internet speed and CPU)

### Run the Web App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start classifying papers!

## 📖 Usage

### 1. Web Interface

The Streamlit app provides four main features:

#### 📝 Text Input
- Paste a paper's title and abstract
- Get instant classification with confidence scores
- View probability distribution across all categories

#### 🔗 ArXiv Lookup
- Enter an ArXiv ID (e.g., `2301.12345`)
- Automatically fetches and classifies the paper
- Displays paper metadata and authors

#### 📁 Batch Upload
- Upload a CSV file with multiple papers
- Classify hundreds of papers at once
- Download results as CSV

#### 📈 Model Performance
- View detailed metrics and comparisons
- See which model was selected
- Compare all tested algorithms

### 2. Python API

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline()

# Classify a single paper
text = "Deep Learning for Natural Language Processing..."
result = pipeline.predict_single(text)
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")

# Classify from ArXiv ID
result = pipeline.predict_from_arxiv_id("2301.12345")
print(f"Title: {result['title']}")
print(f"Category: {result['predicted_category']}")

# Batch predictions
texts = ["paper 1 abstract...", "paper 2 abstract..."]
results = pipeline.predict_batch(texts)
```

## 🏗️ Project Structure

```
Academic-research-AI/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # ArXiv API data fetcher
│   │   ├── data_transformation.py  # NLP preprocessing
│   │   └── model_trainer.py        # AutoML trainer
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Training orchestration
│   │   └── predict_pipeline.py     # Prediction interface
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Logging setup
│   └── exception.py                # Custom exceptions
├── artifacts/                      # Generated files (models, data)
│   ├── data/                       # Cached datasets
│   └── models/                     # Trained models
├── logs/                           # Application logs
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
└── README.md                       # This file
```

## 🔬 Technical Details

### Categories

The system classifies papers into 5 ArXiv categories:

- **cs.AI** - Artificial Intelligence
- **cs.LG** - Machine Learning
- **cs.CL** - Computation and Language (NLP)
- **stat.ML** - Machine Learning (Statistics)
- **cs.CV** - Computer Vision

### Models Tested

The AutoML system evaluates:

1. **Multinomial Naive Bayes** - Fast baseline
2. **Linear SVM** - High-dimensional text classification
3. **Logistic Regression** - Interpretable linear model
4. **Random Forest** - Ensemble method
5. **XGBoost** - Gradient boosting

The best model is automatically selected based on F1-score.

### NLP Pipeline

1. **Text Cleaning**
   - Lowercase conversion
   - URL and email removal
   - Punctuation handling
   - Stopword removal

2. **Feature Extraction**
   - TF-IDF vectorization (5,000 features)
   - Bi-gram support (1-2 word phrases)
   - Feature selection using Chi-squared test

3. **Model Training**
   - 5-fold cross-validation
   - Weighted metrics for class imbalance
   - Automatic hyperparameter tuning

### Performance

**Expected Metrics:**
- Accuracy: 85-92%
- F1 Score: 0.85-0.92
- Training Time: 5-10 minutes
- Prediction Speed: <100ms per paper

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Change categories
categories = ["cs.AI", "cs.LG", "cs.CL", "stat.ML", "cs.CV"]

# Adjust dataset size
max_results_per_category = 500  # Papers per category

# Modify TF-IDF parameters
max_features = 5000
ngram_range = (1, 2)

# Enable/disable models
test_naive_bayes = True
test_svm = True
test_random_forest = True
test_xgboost = True
test_logistic_regression = True
```

## 🧪 Testing

Run individual components:

```bash
# Test data ingestion
python -m src.components.data_ingestion

# Test data transformation
python -m src.components.data_transformation

# Test model training
python -m src.components.model_trainer

# Test prediction
python -m src.pipeline.predict_pipeline
```

## 📊 Model Artifacts

After training, the following files are generated:

```
artifacts/
├── data/
│   ├── raw_papers.csv          # Cached ArXiv data
│   ├── train.csv               # Training set
│   ├── test.csv                # Test set
│   └── val.csv                 # Validation set
└── models/
    ├── best_model.pkl          # Trained model + metadata
    ├── preprocessor.pkl        # TF-IDF vectorizer + label encoder
    ├── metrics.json            # Performance metrics
    └── model_comparison.csv    # All models comparison
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Shivam Bharti**
- GitHub: [@shivam31bharti-arch](https://github.com/shivam31bharti-arch)
- Project: [Academic-research-AI](https://github.com/shivam31bharti-arch/Academic-research-AI)

## 🙏 Acknowledgments

- [ArXiv](https://arxiv.org/) for providing free access to research papers
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Streamlit](https://streamlit.io/) for the web framework
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{academic_research_ai,
  author = {Bharti, Shivam},
  title = {Academic Research AI: Automated Paper Classification},
  year = {2026},
  url = {https://github.com/shivam31bharti-arch/Academic-research-AI}
}
```

---

<div align="center">
  <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>