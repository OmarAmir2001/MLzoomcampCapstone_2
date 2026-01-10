"""
Training module for two-stage hierarchical ML system.

Contains training pipelines for Stage 1, Stage 2, and full pipeline.
"""

import numpy as np
import pandas as pd
from preprocessing import TextPreprocessor, StructuredFeatureEncoder, FeatureCombiner
from models import Stage1Classifier, Stage2Classifier, TwoStagePredictor


def train_stage1(df_train, df_val, model_type='logistic', text_params=None, model_params=None):
    """
    Train Stage 1 classifier (ticket_type prediction).
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        model_type: Type of model ('logistic' or 'svm')
        text_params: Parameters for TextPreprocessor
        model_params: Parameters for Stage1Classifier
        
    Returns:
        Tuple of (model, preprocessors, metrics)
    """
    print("=" * 60)
    print("Training Stage 1: Ticket Type Prediction")
    print("=" * 60)
    
    # Default parameters
    if text_params is None:
        text_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }
    
    if model_params is None:
        model_params = {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    # Define features
    categorical_features = ['product_purchased', 'ticket_channel', 'customer_gender']
    numeric_features = ['customer_age']
    
    # Initialize preprocessors
    print("\n1. Initializing preprocessors...")
    text_preprocessor = TextPreprocessor(**text_params)
    structured_preprocessor = StructuredFeatureEncoder(
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )
    
    # Fit and transform training data
    print("2. Preprocessing training data...")
    text_features_train = text_preprocessor.fit_transform(
        df_train['ticket_subject'],
        df_train['ticket_description']
    )
    structured_features_train = structured_preprocessor.fit_transform(df_train)
    X_train = FeatureCombiner.combine(text_features_train, structured_features_train)
    y_train = df_train['ticket_type']
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Training samples: {len(y_train)}")
    print(f"   Unique ticket types: {y_train.nunique()}")
    
    # Transform validation data
    print("3. Preprocessing validation data...")
    text_features_val = text_preprocessor.transform(
        df_val['ticket_subject'],
        df_val['ticket_description']
    )
    structured_features_val = structured_preprocessor.transform(df_val)
    X_val = FeatureCombiner.combine(text_features_val, structured_features_val)
    y_val = df_val['ticket_type']
    
    # Train model
    print(f"4. Training {model_type} model...")
    model = Stage1Classifier(model_type=model_type, **model_params)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("5. Evaluating on validation set...")
    from evaluation import calculate_metrics
    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val, y_pred, model.classes_)
    
    print(f"\n   Validation Metrics:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   - Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Store preprocessors
    preprocessors = {
        'text': text_preprocessor,
        'structured': structured_preprocessor
    }
    
    print("\n✓ Stage 1 training complete!")
    print("=" * 60)
    
    return model, preprocessors, metrics


def train_stage2(df_train, df_val, stage1_model, preprocessors, 
                 model_type='logistic', model_params=None):
    """
    Train Stage 2 classifier (ticket_priority prediction).
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        stage1_model: Trained Stage1Classifier
        preprocessors: Dictionary with 'text' and 'structured' preprocessors
        model_type: Type of model ('logistic' or 'svm')
        model_params: Parameters for Stage2Classifier
        
    Returns:
        Tuple of (model, metrics)
    """
    print("\n" + "=" * 60)
    print("Training Stage 2: Ticket Priority Prediction")
    print("=" * 60)
    
    # Default parameters
    if model_params is None:
        model_params = {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    # Preprocess training data
    print("\n1. Preprocessing training data...")
    text_features_train = preprocessors['text'].transform(
        df_train['ticket_subject'],
        df_train['ticket_description']
    )
    structured_features_train = preprocessors['structured'].transform(df_train)
    X_train = FeatureCombiner.combine(text_features_train, structured_features_train)
    y_train = df_train['ticket_priority']
    ticket_type_train = df_train['ticket_type']  # Ground-truth for training
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Training samples: {len(y_train)}")
    print(f"   Unique priority levels: {y_train.nunique()}")
    
    # Preprocess validation data
    print("2. Preprocessing validation data...")
    text_features_val = preprocessors['text'].transform(
        df_val['ticket_subject'],
        df_val['ticket_description']
    )
    structured_features_val = preprocessors['structured'].transform(df_val)
    X_val = FeatureCombiner.combine(text_features_val, structured_features_val)
    y_val = df_val['ticket_priority']
    
    # Get predicted ticket_type for validation (simulating inference)
    print("3. Generating Stage 1 predictions for validation...")
    ticket_type_val_pred = stage1_model.predict(X_val)
    
    # Train model (using ground-truth ticket_type)
    print(f"4. Training {model_type} model...")
    model = Stage2Classifier(model_type=model_type, **model_params)
    model.fit(X_train, y_train, ticket_type_train)
    
    # Evaluate on validation set (using predicted ticket_type)
    print("5. Evaluating on validation set (with predicted ticket_type)...")
    from evaluation import calculate_metrics
    y_pred = model.predict(X_val, ticket_type_val_pred)
    metrics = calculate_metrics(y_val, y_pred, model.classes_)
    
    print(f"\n   Validation Metrics:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   - Weighted F1: {metrics['weighted_f1']:.4f}")
    
    print("\n✓ Stage 2 training complete!")
    print("=" * 60)
    
    return model, metrics


def train_full_pipeline(df_train, df_val, config=None):
    """
    Train complete two-stage pipeline.
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (TwoStagePredictor, metrics_dict)
    """
    from utils import get_default_config
    
    if config is None:
        config = get_default_config()
    
    print("\n" + "=" * 60)
    print("Training Full Two-Stage Pipeline")
    print("=" * 60)
    
    # Train Stage 1
    stage1_model, preprocessors, stage1_metrics = train_stage1(
        df_train=df_train,
        df_val=df_val,
        model_type=config['stage1']['model_type'],
        text_params=config['text_preprocessing'],
        model_params=config['stage1']['params']
    )
    
    # Train Stage 2
    stage2_model, stage2_metrics = train_stage2(
        df_train=df_train,
        df_val=df_val,
        stage1_model=stage1_model,
        preprocessors=preprocessors,
        model_type=config['stage2']['model_type'],
        model_params=config['stage2']['params']
    )
    
    # Create combined predictor
    predictor = TwoStagePredictor(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        text_preprocessor=preprocessors['text'],
        structured_preprocessor=preprocessors['structured']
    )
    
    # Combine metrics
    metrics = {
        'stage1': stage1_metrics,
        'stage2': stage2_metrics
    }
    
    print("\n" + "=" * 60)
    print("✓ Full Pipeline Training Complete!")
    print("=" * 60)
    print(f"\nStage 1 (Ticket Type) - Macro F1: {stage1_metrics['macro_f1']:.4f}")
    print(f"Stage 2 (Ticket Priority) - Macro F1: {stage2_metrics['macro_f1']:.4f}")
    print("=" * 60)
    
    return predictor, metrics
