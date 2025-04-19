#!/usr/bin/env python3
"""
Utilities for measuring text similarity and analyzing hallucinations in predictions.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """
    Preprocess text for similarity comparison:
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts using TF-IDF vectors and cosine similarity.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    
    if text1.strip() == "" or text2.strip() == "":
        return 0.0
    
    # Simple exact match gets a perfect score
    if text1.strip().lower() == text2.strip().lower():
        return 1.0
    
    # Preprocess texts
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    try:
        # Fit and transform the texts
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1_processed, text2_processed])
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(cos_sim[0][0])
    except Exception:
        # Fallback for very short texts: character-level similarity
        if len(text1) < 5 or len(text2) < 5:
            # For very short texts, use character overlap ratio
            unique_chars = set(text1_processed + text2_processed)
            if not unique_chars:
                return 0.0
            
            common_chars = set(text1_processed) & set(text2_processed)
            return len(common_chars) / len(unique_chars)
        return 0.0

def detect_hallucination(initial_pred, final_pred, threshold=0.7):
    """
    Detect if a hallucination occurred between initial and final predictions.
    
    Args:
        initial_pred (str): Initial model prediction
        final_pred (str): Final model prediction
        threshold (float): Similarity threshold below which we consider it a hallucination
        
    Returns:
        bool: True if hallucination detected, False otherwise
        float: Similarity score between predictions
    """
    similarity = calculate_similarity(initial_pred, final_pred)
    return similarity < threshold, similarity

def analyze_hallucination_impact(df, initial_pred_col, final_pred_col, accuracy_col=None, threshold=0.7):
    """
    Analyze the impact of hallucinations on model predictions.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
        initial_pred_col (str): Column name for initial predictions
        final_pred_col (str): Column name for final predictions
        accuracy_col (str, optional): Column name for accuracy
        threshold (float): Similarity threshold for hallucination detection
        
    Returns:
        tuple: (DataFrame with analysis results, summary dictionary)
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Add similarity scores
    result_df['similarity'] = result_df.apply(
        lambda row: calculate_similarity(str(row[initial_pred_col]), str(row[final_pred_col])), 
        axis=1
    )
    
    # Detect hallucinations
    result_df['hallucination'] = result_df['similarity'] < threshold
    
    # Calculate summary statistics
    hallucination_count = result_df['hallucination'].sum()
    total_samples = len(result_df)
    hallucination_rate = hallucination_count / total_samples if total_samples > 0 else 0
    
    summary = {
        'hallucination_count': hallucination_count,
        'total_samples': total_samples,
        'hallucination_rate': hallucination_rate
    }
    
    # Analyze impact on accuracy if available
    if accuracy_col and accuracy_col in result_df.columns:
        # Convert accuracy column to numeric if needed
        if result_df[accuracy_col].dtype == 'object':
            try:
                result_df[accuracy_col] = pd.to_numeric(result_df[accuracy_col])
            except ValueError:
                # If conversion fails, skip accuracy analysis
                return result_df, summary
        
        # Calculate accuracy with and without hallucinations
        accuracy_with_hallucination = result_df[result_df['hallucination']].get(accuracy_col, np.nan).mean()
        accuracy_without_hallucination = result_df[~result_df['hallucination']].get(accuracy_col, np.nan).mean()
        
        # Calculate impact (may be NaN if no hallucinations or all hallucinations)
        accuracy_impact = accuracy_without_hallucination - accuracy_with_hallucination
        
        # Add to summary
        summary.update({
            'accuracy_with_hallucination': accuracy_with_hallucination,
            'accuracy_without_hallucination': accuracy_without_hallucination,
            'accuracy_impact': accuracy_impact
        })
    
    return result_df, summary