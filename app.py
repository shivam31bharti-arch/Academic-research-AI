"""
Streamlit Web Application for Academic Research AI
Interactive interface for paper classification.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

from src.pipeline.predict_pipeline import PredictionPipeline
from src.exception import CustomException
from src.config import webapp_config, model_trainer_config

# Page configuration
st.set_page_config(
    page_title=webapp_config.page_title,
    page_icon=webapp_config.page_icon,
    layout=webapp_config.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load prediction pipeline (cached)."""
    try:
        return PredictionPipeline()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running: `python -m src.pipeline.train_pipeline`")
        st.stop()


@st.cache_data
def load_model_metrics():
    """Load model performance metrics."""
    try:
        metrics_path = model_trainer_config.metrics_path
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    except:
        return None


def main():
    # Header
    st.markdown('<div class="main-header">📚 Academic Research AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Paper Classification using NLP & AutoML</div>', unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system automatically classifies academic research papers into categories using:
        - **ArXiv API** for data
        - **NLP** preprocessing
        - **AutoML** model selection
        """)
        
        st.header("📊 Model Info")
        metrics = load_model_metrics()
        if metrics:
            st.metric("Model", metrics.get('model_name', 'Unknown'))
            st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("F1 Score", f"{metrics.get('f1_weighted', 0):.4f}")
        
        st.header("🎯 Categories")
        categories = pipeline.label_encoder.classes_
        for cat in categories:
            st.write(f"• {cat}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "🔗 ArXiv Lookup", "📁 Batch Upload", "📈 Model Performance"])
    
    # Tab 1: Text Input
    with tab1:
        st.header("Classify Research Paper")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            title = st.text_input("Paper Title (optional)", placeholder="Enter paper title...")
            abstract = st.text_area(
                "Paper Abstract *",
                height=200,
                placeholder="Paste the abstract of your research paper here..."
            )
        
        with col2:
            st.write("**Tips:**")
            st.info("""
            - Provide at least the abstract
            - Including the title improves accuracy
            - Longer text generally gives better results
            """)
        
        if st.button("🔍 Classify Paper", type="primary", use_container_width=True):
            if not abstract:
                st.warning("Please enter at least an abstract.")
            else:
                with st.spinner("Analyzing paper..."):
                    # Combine title and abstract
                    text = f"{title} {abstract}" if title else abstract
                    
                    # Make prediction
                    result = pipeline.predict_single(text, return_probabilities=True)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        # Display result
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Category", result['predicted_category'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show probabilities
                        if result['probabilities']:
                            st.subheader("Category Probabilities")
                            
                            # Create DataFrame
                            prob_df = pd.DataFrame([
                                {'Category': cat, 'Probability': prob}
                                for cat, prob in result['probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            
                            # Bar chart
                            fig = px.bar(
                                prob_df,
                                x='Probability',
                                y='Category',
                                orientation='h',
                                color='Probability',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: ArXiv Lookup
    with tab2:
        st.header("Classify Paper from ArXiv")
        
        st.write("Enter an ArXiv paper ID to fetch and classify the paper automatically.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            arxiv_id = st.text_input(
                "ArXiv ID",
                placeholder="e.g., 2301.12345 or arxiv:2301.12345",
                help="You can find the ArXiv ID in the paper's URL"
            )
        
        with col2:
            st.write("")
            st.write("")
            classify_btn = st.button("🔍 Fetch & Classify", use_container_width=True)
        
        if classify_btn:
            if not arxiv_id:
                st.warning("Please enter an ArXiv ID.")
            else:
                with st.spinner(f"Fetching paper {arxiv_id} from ArXiv..."):
                    result = pipeline.predict_from_arxiv_id(arxiv_id)
                    
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Paper info
                        st.subheader("📄 Paper Information")
                        st.write(f"**Title:** {result['title']}")
                        st.write(f"**Authors:** {', '.join(result['authors'])}")
                        if result['published']:
                            st.write(f"**Published:** {result['published'][:10]}")
                        
                        with st.expander("View Abstract"):
                            st.write(result['abstract'])
                        
                        # Prediction
                        st.subheader("🎯 Classification Result")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Category", result['predicted_category'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
    
    # Tab 3: Batch Upload
    with tab3:
        st.header("Batch Classification")
        
        st.write("Upload a CSV file with papers to classify multiple papers at once.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should have columns: 'title' and/or 'abstract'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.write(f"**Loaded {len(df)} papers**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Determine text column
                if 'text' in df.columns:
                    text_col = 'text'
                elif 'abstract' in df.columns and 'title' in df.columns:
                    df['text'] = df['title'] + ' ' + df['abstract']
                    text_col = 'text'
                elif 'abstract' in df.columns:
                    text_col = 'abstract'
                else:
                    st.error("CSV must have 'text', 'abstract', or 'title' column")
                    st.stop()
                
                if st.button("🚀 Classify All Papers", type="primary"):
                    with st.spinner(f"Classifying {len(df)} papers..."):
                        result_df = pipeline.predict_from_dataframe(df, text_col)
                        
                        st.success(f"✅ Classified {len(result_df)} papers!")
                        
                        # Show results
                        st.subheader("Results")
                        st.dataframe(result_df[['predicted_category', 'confidence']], use_container_width=True)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "classification_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # Category distribution
                        st.subheader("Category Distribution")
                        cat_counts = result_df['predicted_category'].value_counts()
                        fig = px.pie(values=cat_counts.values, names=cat_counts.index)
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 4: Model Performance
    with tab4:
        st.header("Model Performance Metrics")
        
        metrics = load_model_metrics()
        
        if metrics:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", metrics.get('model_name', 'N/A'))
            with col2:
                st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with col3:
                st.metric("Test F1 Score", f"{metrics.get('f1_weighted', 0):.4f}")
            with col4:
                st.metric("Val Accuracy", f"{metrics.get('val_accuracy', 0):.2%}")
            
            # Detailed metrics
            st.subheader("Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Test Set Performance**")
                test_metrics = {
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision_weighted', 0),
                    'Recall': metrics.get('recall_weighted', 0),
                    'F1 Score': metrics.get('f1_weighted', 0),
                }
                for metric, value in test_metrics.items():
                    st.write(f"• {metric}: {value:.4f}")
            
            with col2:
                st.write("**Validation Set Performance**")
                val_metrics = {
                    'Accuracy': metrics.get('val_accuracy', 0),
                    'Precision': metrics.get('val_precision_weighted', 0),
                    'Recall': metrics.get('val_recall_weighted', 0),
                    'F1 Score': metrics.get('val_f1_weighted', 0),
                }
                for metric, value in val_metrics.items():
                    st.write(f"• {metric}: {value:.4f}")
            
            # Model comparison
            if 'all_models_comparison' in metrics:
                st.subheader("Model Comparison")
                comparison_df = pd.DataFrame(metrics['all_models_comparison'])
                
                if not comparison_df.empty:
                    # Bar chart
                    fig = px.bar(
                        comparison_df,
                        x='model_name',
                        y='f1_weighted',
                        color='f1_weighted',
                        labels={'model_name': 'Model', 'f1_weighted': 'F1 Score'},
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    with st.expander("View Detailed Comparison"):
                        st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No metrics available. Train the model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Powered by ArXiv API & Scikit-learn</p>
        <p>Created by Shivam Bharti</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
