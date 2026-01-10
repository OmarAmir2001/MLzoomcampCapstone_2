"""
Utility functions for data loading, model serialization, and configuration.
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import os
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load customer support tickets data.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    
    # Clean column names (lowercase and replace spaces)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Clean categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    return df


def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1.0"
    
    # First split: train vs (val + test)
    df_train, df_temp = train_test_split(
        df, 
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=df['ticket_type']  # Stratify by ticket_type
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=df_temp['ticket_type']
    )
    
    return df_train, df_val, df_test


def prepare_features(df, excluded_features=None):
    """
    Prepare DataFrame by excluding specified features.
    
    Args:
        df: Input DataFrame
        excluded_features: List of feature names to exclude
        
    Returns:
        DataFrame with excluded features removed
    """
    if excluded_features is None:
        excluded_features = ['ticket_priority', 'ticket_status', 'date_of_purchase']
    
    # Remove excluded features if they exist
    cols_to_drop = [col for col in excluded_features if col in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


def save_model(model, filepath):
    """
    Save model to disk.
    
    Args:
        model: Model object to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load model from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model object
    """
    return joblib.load(filepath)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config():
    """
    Get default configuration for the ML pipeline.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'data': {
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            'random_state': 42
        },
        'text_preprocessing': {
            'max_features': 5000,
            'ngram_range': (1, 2),  # Fixed: tuple instead of list
            'min_df': 2,
            'max_df': 0.95
        },
        'structured_features': {
            'categorical': ['product_purchased', 'ticket_channel', 'customer_gender'],
            'numeric': ['customer_age'],
            'excluded': ['ticket_priority', 'ticket_status', 'date_of_purchase']
        },
        'stage1': {
            'model_type': 'logistic',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        'stage2': {
            'model_type': 'logistic',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
        }
    }


def print_data_info(df, name="Dataset"):
    """
    Print information about dataset.
    
    Args:
        df: DataFrame
        name: Name of dataset for display
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    if 'ticket_type' in df.columns:
        print(f"\n  Ticket Type Distribution:")
        print(df['ticket_type'].value_counts())
    
    if 'ticket_priority' in df.columns:
        print(f"\n  Ticket Priority Distribution:")
        print(df['ticket_priority'].value_counts())
