"""
Data preprocessing module for two-stage hierarchical ML system.

This module handles all feature engineering including:
- Text preprocessing using TF-IDF
- Structured feature encoding (one-hot for categorical, scaling for numeric)
- Feature combination
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib


class TextPreprocessor:
    """
    Preprocesses text features using TF-IDF vectorization.
    
    Combines ticket_subject and ticket_description into a single text field
    and applies TF-IDF transformation.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95):
        """
        Initialize TextPreprocessor.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents='unicode'
        )
        self.is_fitted = False
    
    def _combine_text(self, ticket_subject, ticket_description):
        """
        Combine ticket_subject and ticket_description into single text.
        
        Args:
            ticket_subject: Series or array of ticket subjects
            ticket_description: Series or array of ticket descriptions
            
        Returns:
            Combined text as list of strings
        """
        combined = []
        for subj, desc in zip(ticket_subject, ticket_description):
            # Combine with space separator
            text = f"{subj} {desc}"
            combined.append(text)
        return combined
    
    def fit(self, ticket_subject, ticket_description):
        """
        Fit TF-IDF vectorizer on combined text.
        
        Args:
            ticket_subject: Series or array of ticket subjects
            ticket_description: Series or array of ticket descriptions
            
        Returns:
            self
        """
        combined_text = self._combine_text(ticket_subject, ticket_description)
        self.vectorizer.fit(combined_text)
        self.is_fitted = True
        return self
    
    def transform(self, ticket_subject, ticket_description):
        """
        Transform text using fitted TF-IDF vectorizer.
        
        Args:
            ticket_subject: Series or array of ticket subjects
            ticket_description: Series or array of ticket descriptions
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before transform")
        
        combined_text = self._combine_text(ticket_subject, ticket_description)
        return self.vectorizer.transform(combined_text)
    
    def fit_transform(self, ticket_subject, ticket_description):
        """
        Fit and transform text in one step.
        
        Args:
            ticket_subject: Series or array of ticket subjects
            ticket_description: Series or array of ticket descriptions
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        combined_text = self._combine_text(ticket_subject, ticket_description)
        tfidf_features = self.vectorizer.fit_transform(combined_text)
        self.is_fitted = True
        return tfidf_features
    
    def get_feature_names(self):
        """Get feature names from TF-IDF vectorizer."""
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before getting feature names")
        return self.vectorizer.get_feature_names_out()


class StructuredFeatureEncoder:
    """
    Encodes structured features (categorical and numeric).
    
    - Categorical features: One-hot encoding
    - Numeric features: Standard scaling
    """
    
    def __init__(self, categorical_features, numeric_features):
        """
        Initialize StructuredFeatureEncoder.
        
        Args:
            categorical_features: List of categorical feature names
            numeric_features: List of numeric feature names
        """
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.categorical_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.numeric_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, df):
        """
        Fit encoders on structured features.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            self
        """
        # Fit categorical encoder
        if self.categorical_features:
            self.categorical_encoder.fit(df[self.categorical_features])
        
        # Fit numeric scaler
        if self.numeric_features:
            self.numeric_scaler.fit(df[self.numeric_features])
        
        self.is_fitted = True
        return self
    
    def transform(self, df):
        """
        Transform structured features using fitted encoders.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            Sparse matrix of encoded features
        """
        if not self.is_fitted:
            raise ValueError("StructuredFeatureEncoder must be fitted before transform")
        
        features = []
        
        # Transform categorical features
        if self.categorical_features:
            cat_encoded = self.categorical_encoder.transform(df[self.categorical_features])
            features.append(cat_encoded)
        
        # Transform numeric features
        if self.numeric_features:
            num_scaled = self.numeric_scaler.transform(df[self.numeric_features])
            # Convert to sparse matrix
            num_sparse = csr_matrix(num_scaled)
            features.append(num_sparse)
        
        # Combine all features
        if len(features) > 1:
            return hstack(features)
        elif len(features) == 1:
            return features[0]
        else:
            raise ValueError("No features to transform")
    
    def fit_transform(self, df):
        """
        Fit and transform structured features in one step.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            Sparse matrix of encoded features
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self):
        """Get feature names after encoding."""
        if not self.is_fitted:
            raise ValueError("StructuredFeatureEncoder must be fitted before getting feature names")
        
        feature_names = []
        
        # Categorical feature names
        if self.categorical_features:
            cat_names = self.categorical_encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_names)
        
        # Numeric feature names
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        return feature_names


class FeatureCombiner:
    """
    Combines text and structured features into final feature matrix.
    """
    
    @staticmethod
    def combine(text_features, structured_features):
        """
        Combine text and structured features.
        
        Args:
            text_features: Sparse matrix of text features (TF-IDF)
            structured_features: Sparse matrix of structured features
            
        Returns:
            Combined sparse matrix
        """
        return hstack([text_features, structured_features])


def save_preprocessors(preprocessors, filepath):
    """
    Save preprocessors to disk.
    
    Args:
        preprocessors: Dictionary of preprocessor objects
        filepath: Path to save file
    """
    joblib.dump(preprocessors, filepath)


def load_preprocessors(filepath):
    """
    Load preprocessors from disk.
    
    Args:
        filepath: Path to saved preprocessors
        
    Returns:
        Dictionary of preprocessor objects
    """
    return joblib.load(filepath)
