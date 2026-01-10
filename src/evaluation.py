"""
Evaluation module for model performance assessment.

Contains functions for calculating metrics, generating confusion matrices,
and visualizing results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (optional)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
    }
    
    # Per-class metrics
    if labels is not None:
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)
        metrics['per_class_f1'] = dict(zip(labels, per_class_f1))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", 
                          figsize=(10, 8), save_path=None):
    """
    Plot confusion matrix with nice formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        title: Plot title
        figsize: Figure size tuple
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )
    
    # Add counts as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                          ha="center", va="center", color="gray", fontsize=8)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def print_classification_report(y_true, y_pred, labels=None, title="Classification Report"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (optional)
        title: Report title
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("=" * 60)


def evaluate_stage1(model, df, preprocessors, verbose=True):
    """
    Evaluate Stage 1 model (ticket_type prediction).
    
    Args:
        model: Trained Stage1Classifier
        df: DataFrame to evaluate on
        preprocessors: Dictionary with 'text' and 'structured' preprocessors
        verbose: Whether to print results
        
    Returns:
        Dictionary with metrics and predictions
    """
    from preprocessing import FeatureCombiner
    
    # Preprocess features
    text_features = preprocessors['text'].transform(
        df['ticket_subject'],
        df['ticket_description']
    )
    structured_features = preprocessors['structured'].transform(df)
    X = FeatureCombiner.combine(text_features, structured_features)
    
    # Get predictions
    y_true = df['ticket_type']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, model.classes_)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 1 Evaluation: Ticket Type Prediction")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        print_classification_report(y_true, y_pred, model.classes_, 
                                   "Stage 1 Classification Report")
    
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'labels': model.classes_
    }


def evaluate_stage2(model, df, stage1_model, preprocessors, verbose=True):
    """
    Evaluate Stage 2 model (ticket_priority prediction).
    
    Args:
        model: Trained Stage2Classifier
        df: DataFrame to evaluate on
        stage1_model: Trained Stage1Classifier (for predicting ticket_type)
        preprocessors: Dictionary with 'text' and 'structured' preprocessors
        verbose: Whether to print results
        
    Returns:
        Dictionary with metrics and predictions
    """
    from preprocessing import FeatureCombiner
    
    # Preprocess features
    text_features = preprocessors['text'].transform(
        df['ticket_subject'],
        df['ticket_description']
    )
    structured_features = preprocessors['structured'].transform(df)
    X = FeatureCombiner.combine(text_features, structured_features)
    
    # Get Stage 1 predictions (simulating inference)
    ticket_type_pred = stage1_model.predict(X)
    
    # Get Stage 2 predictions
    y_true = df['ticket_priority']
    y_pred = model.predict(X, ticket_type_pred)
    y_proba = model.predict_proba(X, ticket_type_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, model.classes_)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 2 Evaluation: Ticket Priority Prediction")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        print_classification_report(y_true, y_pred, model.classes_,
                                   "Stage 2 Classification Report")
    
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'labels': model.classes_,
        'ticket_type_pred': ticket_type_pred
    }


def evaluate_full_pipeline(predictor, df, verbose=True):
    """
    Evaluate complete two-stage pipeline.
    
    Args:
        predictor: TwoStagePredictor instance
        df: DataFrame to evaluate on
        verbose: Whether to print results
        
    Returns:
        Dictionary with metrics for both stages
    """
    # Get predictions
    predictions = predictor.predict(df)
    
    # Evaluate Stage 1
    y_true_type = df['ticket_type']
    y_pred_type = predictions['ticket_type']
    stage1_metrics = calculate_metrics(y_true_type, y_pred_type, 
                                      predictor.stage1_model.classes_)
    
    # Evaluate Stage 2
    y_true_priority = df['ticket_priority']
    y_pred_priority = predictions['ticket_priority']
    stage2_metrics = calculate_metrics(y_true_priority, y_pred_priority,
                                      predictor.stage2_model.classes_)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Full Pipeline Evaluation")
        print("=" * 60)
        print("\nStage 1 (Ticket Type):")
        print(f"  Accuracy: {stage1_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {stage1_metrics['macro_f1']:.4f}")
        
        print("\nStage 2 (Ticket Priority):")
        print(f"  Accuracy: {stage2_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {stage2_metrics['macro_f1']:.4f}")
        print("=" * 60)
    
    return {
        'stage1': {
            'metrics': stage1_metrics,
            'y_true': y_true_type,
            'y_pred': y_pred_type,
            'labels': predictor.stage1_model.classes_
        },
        'stage2': {
            'metrics': stage2_metrics,
            'y_true': y_true_priority,
            'y_pred': y_pred_priority,
            'labels': predictor.stage2_model.classes_
        }
    }


def create_evaluation_report(predictor, df_test, output_dir='reports'):
    """
    Create comprehensive evaluation report with plots.
    
    Args:
        predictor: TwoStagePredictor instance
        df_test: Test DataFrame
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with all evaluation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Creating Comprehensive Evaluation Report")
    print("=" * 60)
    
    # Evaluate full pipeline
    results = evaluate_full_pipeline(predictor, df_test, verbose=True)
    
    # Plot Stage 1 confusion matrix
    print("\nGenerating Stage 1 confusion matrix...")
    fig1 = plot_confusion_matrix(
        results['stage1']['y_true'],
        results['stage1']['y_pred'],
        results['stage1']['labels'],
        title="Stage 1: Ticket Type Prediction",
        save_path=os.path.join(output_dir, 'stage1_confusion_matrix.png')
    )
    plt.close(fig1)
    
    # Plot Stage 2 confusion matrix
    print("Generating Stage 2 confusion matrix...")
    fig2 = plot_confusion_matrix(
        results['stage2']['y_true'],
        results['stage2']['y_pred'],
        results['stage2']['labels'],
        title="Stage 2: Ticket Priority Prediction",
        save_path=os.path.join(output_dir, 'stage2_confusion_matrix.png')
    )
    plt.close(fig2)
    
    # Print classification reports
    print_classification_report(
        results['stage1']['y_true'],
        results['stage1']['y_pred'],
        results['stage1']['labels'],
        "Stage 1: Ticket Type Classification Report"
    )
    
    print_classification_report(
        results['stage2']['y_true'],
        results['stage2']['y_pred'],
        results['stage2']['labels'],
        "Stage 2: Ticket Priority Classification Report"
    )
    
    print(f"\n✓ Evaluation report complete! Plots saved to: {output_dir}")
    
    return results
