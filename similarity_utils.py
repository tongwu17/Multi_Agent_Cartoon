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

def calculate_similarity(text1, text2, weights=None):
    """
    Calculate similarity between two texts using TF-IDF vectors and cosine similarity.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        weights (dict, optional): Weights for different similarity components
            - 'exact_match': Weight for exact match (default: 0.3)
            - 'tfidf': Weight for TF-IDF similarity (default: 0.6)
            - 'char_overlap': Weight for character overlap (default: 0.1)
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Set default weights if not provided
    if weights is None:
        weights = {
            'exact_match': 0.3,
            'tfidf': 0.6,
            'char_overlap': 0.1
        }
    
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
        tfidf_sim = float(cos_sim[0][0])
        
        # Calculate character overlap similarity for more robust comparison
        unique_chars = set(text1_processed + text2_processed)
        common_chars = set(text1_processed) & set(text2_processed)
        char_overlap = len(common_chars) / len(unique_chars) if unique_chars else 0.0
        
        # Combine similarities using weights
        final_sim = (
            weights['exact_match'] * (1.0 if text1.strip().lower() == text2.strip().lower() else 0.0) +
            weights['tfidf'] * tfidf_sim +
            weights['char_overlap'] * char_overlap
        )
        
        return min(1.0, final_sim)  # Cap at 1.0
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

def analyze_agent_consistency(visual_desc, halluc_answer, initial_answer, threshold_high=0.7, threshold_low=0.4):
    """
    Analyze consistency between visual description and hallucination agent's answer.
    This helps determine if visual and hallucination agents are in agreement.
    
    Args:
        visual_desc (str): Visual description from visual agent
        halluc_answer (str): Answer from hallucination agent
        initial_answer (str): Initial answer from language agent
        threshold_high (float): High consistency threshold
        threshold_low (float): Low consistency threshold
        
    Returns:
        dict: Analysis results containing:
            - consistency_score: Float between 0-1
            - is_consistent: Boolean indicating if agents are consistent
            - confidence: 'high', 'medium', or 'low'
            - recommend_agent: Which agent's answer to prioritize ('visual', 'hallucination', 'language', or 'combined')
    """
    # Check if the hallucination agent's answer appears in the visual description
    # This is a simple indicator that the hallucination agent's answer
    # is grounded in the visual content
    halluc_in_visual = halluc_answer.lower() in visual_desc.lower()
    
    # Calculate similarity between the initial answer and hallucination agent's answer
    lang_halluc_sim = calculate_similarity(initial_answer, halluc_answer)
    
    # Analyze visual description for evidence supporting each answer
    visual_supports_initial = False
    visual_supports_halluc = False
    
    # Check if visual description contains keywords related to the answers
    initial_keywords = initial_answer.lower().split()
    halluc_keywords = halluc_answer.lower().split()
    
    # Count how many keywords from each answer appear in the visual description
    initial_matches = sum(1 for word in initial_keywords if word in visual_desc.lower())
    halluc_matches = sum(1 for word in halluc_keywords if word in visual_desc.lower())
    
    # Normalize by the number of keywords
    initial_match_ratio = initial_matches / len(initial_keywords) if initial_keywords else 0
    halluc_match_ratio = halluc_matches / len(halluc_keywords) if halluc_keywords else 0
    
    # Determine if visual description supports each answer
    visual_supports_initial = initial_match_ratio > 0.3
    visual_supports_halluc = halluc_match_ratio > 0.3
    
    # Calculate overall consistency score
    # Higher weight to hallucination in visual (most important alignment)
    consistency_score = 0.5 * int(halluc_in_visual) + 0.3 * halluc_match_ratio + 0.2 * lang_halluc_sim
    
    # Determine if agents are consistent
    is_consistent = consistency_score >= threshold_high
    
    # Determine confidence level
    if consistency_score >= threshold_high:
        confidence = 'high'
    elif consistency_score >= threshold_low:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    # Recommend which agent's answer to prioritize
    if halluc_in_visual and visual_supports_halluc:
        # Strong agreement between visual and hallucination
        recommend_agent = 'hallucination'
    elif visual_supports_initial and not visual_supports_halluc:
        # Visual supports language but not hallucination
        recommend_agent = 'language'
    elif visual_supports_halluc and not visual_supports_initial:
        # Visual supports hallucination but not language
        recommend_agent = 'hallucination'
    elif lang_halluc_sim > 0.7:
        # Language and hallucination agents mostly agree
        recommend_agent = 'combined'
    elif confidence == 'low':
        # Low confidence, default to hallucination agent which has shown better performance
        recommend_agent = 'hallucination'
    else:
        # Default case
        recommend_agent = 'combined'
    
    return {
        'consistency_score': consistency_score,
        'is_consistent': is_consistent,
        'confidence': confidence,
        'recommend_agent': recommend_agent,
        'visual_supports_initial': visual_supports_initial,
        'visual_supports_halluc': visual_supports_halluc,
        'hallucination_in_visual': halluc_in_visual,
        'language_halluc_similarity': lang_halluc_sim
    }

def agent_coordinator(visual_desc, initial_answer, halluc_answer, question_type=None):
    """
    Coordinate decisions between visual, language and hallucination agents
    based on a sequential processing flow rather than weighted voting.
    
    Args:
        visual_desc (str): Visual description from visual agent
        initial_answer (str): Initial answer from language agent
        halluc_answer (str): Answer from hallucination agent
        question_type (str, optional): Type of question (visual, factual, reasoning, etc.)
        
    Returns:
        dict: Coordination result containing:
            - final_answer: The final coordinated answer
            - confidence: Confidence level in the answer
            - rationale: Explanation of the decision process
    """
    # If question_type is not provided, try to infer it
    if question_type is None:
        question_type = infer_question_type(visual_desc, initial_answer, halluc_answer)
    
    # Calculate consistency metrics for analysis
    visual_lang_consistency = detect_content_overlap(visual_desc, initial_answer)
    visual_halluc_consistency = detect_content_overlap(visual_desc, halluc_answer)
    lang_halluc_consistency = calculate_similarity(initial_answer, halluc_answer)
    
    # Default to using hallucination agent's answer as it showed best performance in ablation study
    final_answer = halluc_answer
    confidence = "MEDIUM"
    rationale = "Using hallucination agent's verified answer by default"
    
    # Sequential decision process based on question type and consistency patterns
    if question_type == 'visual':
        # For visual questions, visual evidence is critical
        if visual_halluc_consistency > 0.6:
            # Hallucination agent's answer is well-supported by visual description
            final_answer = halluc_answer
            confidence = "HIGH"
            rationale = "Hallucination agent's answer is strongly supported by visual evidence"
        elif visual_lang_consistency > 0.6 and visual_lang_consistency > visual_halluc_consistency:
            # Visual evidence supports language agent's answer more than hallucination agent's correction
            final_answer = initial_answer
            confidence = "MEDIUM"
            rationale = "Visual evidence supports language agent's original answer more than the correction"
        else:
            # Default to hallucination agent with medium confidence
            final_answer = halluc_answer
            confidence = "MEDIUM"
            rationale = "Using hallucination agent's verification for visual question with moderate confidence"
    
    elif question_type == 'factual':
        # For factual questions, hallucination detection is most important
        if lang_halluc_consistency > 0.8:
            # Language and hallucination agents agree strongly
            final_answer = halluc_answer  # Use hallucination agent's answer for consistency
            confidence = "HIGH"
            rationale = "Language and hallucination agents strongly agree on factual answer"
        elif halluc_answer.lower() in visual_desc.lower():
            # Direct evidence in visual description supports hallucination agent
            final_answer = halluc_answer
            confidence = "HIGH"
            rationale = "Visual description directly contains the hallucination agent's answer"
        else:
            # Default to hallucination agent with high confidence for factual questions
            final_answer = halluc_answer
            confidence = "HIGH"
            rationale = "Using hallucination agent's verification for factual question"
    
    else:  # reasoning or other question types
        # For reasoning questions, consider all evidence
        if lang_halluc_consistency > 0.7:
            # Language and hallucination agents mostly agree
            final_answer = halluc_answer
            confidence = "HIGH"
            rationale = "Language and hallucination agents agree on reasoning question"
        elif visual_halluc_consistency > 0.5 and visual_lang_consistency < 0.3:
            # Visual evidence supports hallucination but not language
            final_answer = halluc_answer
            confidence = "HIGH"
            rationale = "Visual evidence supports hallucination agent's answer for reasoning question"
        elif visual_lang_consistency > 0.5 and visual_halluc_consistency < 0.3:
            # Visual evidence supports language but not hallucination
            final_answer = initial_answer
            confidence = "MEDIUM"
            rationale = "Visual evidence supports language agent's original answer"
        else:
            # Default to hallucination agent with medium confidence
            final_answer = halluc_answer
            confidence = "MEDIUM" 
            rationale = "Using hallucination agent's verification for reasoning question"
    
    # Special case: when initial and hallucination answers are identical, high confidence
    if initial_answer.lower().strip() == halluc_answer.lower().strip():
        confidence = "HIGH"
        rationale = "Language and hallucination agents produced identical answers"
    
    # Special case: when hallucination agent detected very low confidence
    if halluc_answer.lower().startswith("[low confidence]"):
        confidence = "LOW"
        # Remove the prefix for the final answer
        final_answer = halluc_answer.lower().replace("[low confidence]", "").strip()
        rationale = "Hallucination agent expressed low confidence in its verification"
    
    return {
        'final_answer': final_answer,
        'confidence': confidence,
        'rationale': rationale,
        'consistencies': {
            'visual_language': visual_lang_consistency,
            'visual_hallucination': visual_halluc_consistency,
            'language_hallucination': lang_halluc_consistency
        }
    }

def infer_question_type(visual_desc, initial_answer, halluc_answer):
    """
    Infer the question type based on the agents' responses.
    
    Args:
        visual_desc (str): Visual description from visual agent
        initial_answer (str): Initial answer from language agent
        halluc_answer (str): Answer from hallucination agent
        
    Returns:
        str: Inferred question type ('visual', 'factual', or 'reasoning')
    """
    # Check for visual question indicators
    visual_keywords = ['color', 'wearing', 'look', 'position', 'left', 'right', 
                      'background', 'front', 'behind', 'above', 'below']
                      
    # Count visual keywords in the visual description
    visual_keyword_count = sum(1 for keyword in visual_keywords 
                             if keyword in visual_desc.lower())
    
    # If many visual keywords are present, it's likely a visual question
    if visual_keyword_count >= 3:
        return 'visual'
    
    # Check if the answer is a simple fact (short answers are often factual)
    if len(initial_answer.split()) <= 2 and len(halluc_answer.split()) <= 2:
        return 'factual'
    
    # Default to reasoning for more complex questions
    return 'reasoning'

def detect_content_overlap(text1, text2):
    """
    Detect if content from text2 appears in text1, providing a 
    measure of content overlap rather than just similarity.
    
    Args:
        text1 (str): First text (usually longer, like visual description)
        text2 (str): Second text (usually shorter, like an answer)
        
    Returns:
        float: Overlap score between 0 and 1
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Direct containment is a strong signal
    if text2 in text1:
        return 1.0
    
    # Check for keyword overlap
    words2 = set(text2.split())
    if not words2:
        return 0.0
    
    # Count how many words from text2 appear in text1
    matches = sum(1 for word in words2 if word in text1)
    overlap_ratio = matches / len(words2)
    
    return overlap_ratio