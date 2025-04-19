#!/usr/bin/env python3
"""
Analyze hallucinations in ablation study results.
This script finds differences between initial and final predictions to detect hallucinations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from similarity_utils import calculate_similarity, analyze_hallucination_impact

def load_csv_file(filepath):
    """Load CSV file with predictions."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_prediction_columns(df):
    """Find columns containing initial and final predictions."""
    initial_pred_cols = [col for col in df.columns if 'initial' in col.lower() and 'predict' in col.lower()]
    final_pred_cols = [col for col in df.columns if 'predict' in col.lower() and 'initial' not in col.lower()]
    
    if initial_pred_cols and final_pred_cols:
        return initial_pred_cols[0], final_pred_cols[0]
    return None, None

def analyze_file(filepath, verbose=True):
    """Analyze hallucinations in a single CSV file."""
    if verbose:
        print(f"\nAnalyzing hallucination impact in {os.path.basename(filepath)}:")
    
    df = load_csv_file(filepath)
    if df is None:
        return None, None
    
    # Find prediction columns
    initial_pred_col, final_pred_col = find_prediction_columns(df)
    
    if initial_pred_col is None or final_pred_col is None:
        print(f"No prediction columns found in {os.path.basename(filepath)}")
        return None, None
    
    if verbose:
        print(f"Using initial prediction column: {initial_pred_col}")
        print(f"Using final prediction column: {final_pred_col}")
    
    # Check if an accuracy column exists
    accuracy_col = None
    for col in df.columns:
        if 'accuracy' in col.lower():
            accuracy_col = col
            break
    
    try:
        result_df, summary = analyze_hallucination_impact(
            df, 
            initial_pred_col, 
            final_pred_col, 
            accuracy_col
        )
        
        if verbose:
            print(f"Hallucinations detected: {summary['hallucination_count']} out of {summary['total_samples']} samples")
            print(f"Hallucination rate: {summary['hallucination_rate']:.2%}")
            
            if 'accuracy_impact' in summary:
                print(f"Accuracy with hallucinations: {summary['accuracy_with_hallucination']:.2f}")
                print(f"Accuracy without hallucinations: {summary['accuracy_without_hallucination']:.2f}")
                print(f"Accuracy impact: {summary['accuracy_impact']:.2f}")
        
        return result_df, summary
        
    except Exception as e:
        print(f"Error analyzing {os.path.basename(filepath)}: {str(e)}")
        return None, None

def plot_hallucination_summary(summaries, filenames):
    """Create visualization of hallucination analysis results across files."""
    if not summaries:
        print("No valid summaries to plot")
        return
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Dataset': [os.path.splitext(os.path.basename(f))[0] for f in filenames],
        'Hallucination Rate': [s['hallucination_rate'] for s in summaries],
        'Accuracy Impact': [s.get('accuracy_impact', np.nan) for s in summaries]
    })
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot hallucination rates
    sns.barplot(x='Dataset', y='Hallucination Rate', data=plot_data, ax=ax1)
    ax1.set_title('Hallucination Rate by Dataset')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Plot accuracy impact where available
    accuracy_data = plot_data.dropna(subset=['Accuracy Impact'])
    if not accuracy_data.empty:
        sns.barplot(x='Dataset', y='Accuracy Impact', data=accuracy_data, ax=ax2)
        ax2.set_title('Accuracy Impact of Hallucinations')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No accuracy impact data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('hallucination_analysis.png')
    print("Saved visualization to hallucination_analysis.png")
    plt.close()

def save_detailed_results(result_dfs, filenames):
    """Save detailed hallucination analysis results to CSV files."""
    for df, filename in zip(result_dfs, filenames):
        if df is not None:
            output_filename = os.path.join(
                'results', 
                'analysis',
                f"hallucination_analysis_{os.path.basename(filename)}"
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_filename, index=False)
            print(f"Saved detailed analysis to {output_filename}")

def main():
    """Main entry point for hallucination analysis."""
    # Find all ablation CSV files
    ablation_dir = os.path.join('results', 'ablation')
    if not os.path.exists(ablation_dir):
        print(f"Error: Directory not found: {ablation_dir}")
        return
        
    csv_files = [os.path.join(ablation_dir, f) for f in os.listdir(ablation_dir) 
                if f.endswith('.csv') and 'hallucination' in f.lower()]
    
    if not csv_files:
        print("No hallucination CSV files found in results/ablation/")
        return
    
    print(f"Found {len(csv_files)} CSV files with hallucination data")
    
    # Analyze each file
    result_dfs = []
    summaries = []
    valid_files = []
    
    for filepath in csv_files:
        result_df, summary = analyze_file(filepath)
        if result_df is not None and summary is not None:
            result_dfs.append(result_df)
            summaries.append(summary)
            valid_files.append(filepath)
    
    if not summaries:
        print("No files could be successfully analyzed")
        return
    
    # Generate visualizations
    plot_hallucination_summary(summaries, valid_files)
    
    # Save detailed results
    save_detailed_results(result_dfs, valid_files)
    
    print("\nHallucination Analysis Summary:")
    print("=" * 50)
    for filename, summary in zip(valid_files, summaries):
        print(f"\n{os.path.basename(filename)}:")
        print(f"  - Hallucination rate: {summary['hallucination_rate']:.2%}")
        if 'accuracy_impact' in summary:
            print(f"  - Accuracy impact: {summary['accuracy_impact']:.4f}")
    
    print("\nAnalysis complete! Check hallucination_analysis.png for visualization.")

if __name__ == "__main__":
    main()