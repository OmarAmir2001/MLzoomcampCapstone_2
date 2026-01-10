"""
Model classes for two-stage hierarchical ML system.

This module contains:
- Stage1Classifier: Predicts ticket_type
- Stage2Classifier: Predicts ticket_priority
- TwoStagePredictor: Combined inference pipeline
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
from scipy.sparse import hstack, csr_matrix


class Stage1Classifier:
    """
    Stage 1 classifier for predicting ticket_type.
    
    Uses text features (TF-IDF) and structured features to predict ticket type.
    """
    
    def __init__(self, model_type='logistic', **kwargs):
        """
        Initialize Stage 1 classifier.
        
        Args:
            model_type: Type of model ('logistic' or 'svm')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = kwargs
        
        # Set default parameters
        if 'class_weight' not in self.model_params:
            self.model_params['class_weight'] = 'balanced'
        if 'random_state' not in self.model_params:
            self.model_params['random_state'] = 42
        
        # Initialize model
        if model_type == 'logistic':
            if 'max_iter' not in self.model_params:
                self.model_params['max_iter'] = 1000
            self.model = LogisticRegression(**self.model_params)
        elif model_type == 'svm':
            if 'max_iter' not in self.model_params:
                self.model_params['max_iter'] = 1000
            # LinearSVC doesn't support predict_proba, so we'll calibrate it
            base_model = LinearSVC(**self.model_params)
            self.model = CalibratedClassifierCV(base_model, cv=3)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.is_fitted = False
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Fit the Stage 1 classifier.
        
        Args:
            X: Feature matrix (sparse or dense)
            y: Target labels (ticket_type)
            
        Returns:
            self
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict ticket_type.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load model from disk."""
        return joblib.load(filepath)


class Stage2Classifier:
    """
    Stage 2 classifier for predicting ticket_priority.
    
    Uses text features, structured features, AND ticket_type to predict priority.
    During training: uses ground-truth ticket_type
    During inference: uses predicted ticket_type from Stage 1
    """
    
    def __init__(self, model_type='logistic', **kwargs):
        """
        Initialize Stage 2 classifier.
        
        Args:
            model_type: Type of model ('logistic' or 'svm')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = kwargs
        
        # Set default parameters
        if 'class_weight' not in self.model_params:
            self.model_params['class_weight'] = 'balanced'
        if 'random_state' not in self.model_params:
            self.model_params['random_state'] = 42
        
        # Initialize model
        if model_type == 'logistic':
            if 'max_iter' not in self.model_params:
                self.model_params['max_iter'] = 1000
            self.model = LogisticRegression(**self.model_params)
        elif model_type == 'svm':
            if 'max_iter' not in self.model_params:
                self.model_params['max_iter'] = 1000
            base_model = LinearSVC(**self.model_params)
            self.model = CalibratedClassifierCV(base_model, cv=3)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.is_fitted = False
        self.classes_ = None
        self.ticket_type_encoder = None
    
    def _encode_ticket_type(self, ticket_type_labels, fit=False):
        """
        One-hot encode ticket_type labels.
        
        Args:
            ticket_type_labels: Array of ticket_type labels
            fit: Whether to fit the encoder
            
        Returns:
            Sparse matrix of encoded ticket_type
        """
        from sklearn.preprocessing import OneHotEncoder
        
        if fit:
            self.ticket_type_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
            encoded = self.ticket_type_encoder.fit_transform(
                np.array(ticket_type_labels).reshape(-1, 1)
            )
        else:
            if self.ticket_type_encoder is None:
                raise ValueError("Encoder not fitted")
            encoded = self.ticket_type_encoder.transform(
                np.array(ticket_type_labels).reshape(-1, 1)
            )
        
        return encoded
    
    def fit(self, X, y, ticket_type_labels):
        """
        Fit the Stage 2 classifier.
        
        Args:
            X: Feature matrix (text + structured features)
            y: Target labels (ticket_priority)
            ticket_type_labels: Ground-truth ticket_type labels
            
        Returns:
            self
        """
        # Encode ticket_type and add to features
        ticket_type_encoded = self._encode_ticket_type(ticket_type_labels, fit=True)
        X_combined = hstack([X, ticket_type_encoded])
        
        # Fit model
        self.model.fit(X_combined, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self
    
    def predict(self, X, predicted_ticket_type):
        """
        Predict ticket_priority using predicted ticket_type.
        
        Args:
            X: Feature matrix (text + structured features)
            predicted_ticket_type: Predicted ticket_type from Stage 1
            
        Returns:
            Predicted priority labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Encode predicted ticket_type and add to features
        ticket_type_encoded = self._encode_ticket_type(predicted_ticket_type, fit=False)
        X_combined = hstack([X, ticket_type_encoded])
        
        return self.model.predict(X_combined)
    
    def predict_proba(self, X, predicted_ticket_type):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            predicted_ticket_type: Predicted ticket_type from Stage 1
            
        Returns:
            Probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Encode predicted ticket_type and add to features
        ticket_type_encoded = self._encode_ticket_type(predicted_ticket_type, fit=False)
        X_combined = hstack([X, ticket_type_encoded])
        
        return self.model.predict_proba(X_combined)
    
    def save(self, filepath):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load model from disk."""
        return joblib.load(filepath)


class TwoStagePredictor:
    """
    Combined two-stage prediction pipeline.
    
    Orchestrates Stage 1 and Stage 2 predictions with preprocessing.
    """
    
    def __init__(self, stage1_model, stage2_model, text_preprocessor, structured_preprocessor):
        """
        Initialize two-stage predictor.
        
        Args:
            stage1_model: Fitted Stage1Classifier
            stage2_model: Fitted Stage2Classifier
            text_preprocessor: Fitted TextPreprocessor
            structured_preprocessor: Fitted StructuredFeatureEncoder
        """
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.text_preprocessor = text_preprocessor
        self.structured_preprocessor = structured_preprocessor
    
    def predict(self, df):
        """
        End-to-end prediction for both ticket_type and ticket_priority.
        
        Args:
            df: DataFrame with columns:
                - ticket_subject
                - ticket_description
                - product_purchased
                - ticket_channel
                - customer_gender
                - customer_age
                
        Returns:
            DataFrame with predicted ticket_type and ticket_priority
        """
        # Preprocess features
        text_features = self.text_preprocessor.transform(
            df['ticket_subject'], 
            df['ticket_description']
        )
        structured_features = self.structured_preprocessor.transform(df)
        
        # Combine features
        from preprocessing import FeatureCombiner
        X = FeatureCombiner.combine(text_features, structured_features)
        
        # Stage 1: Predict ticket_type
        predicted_ticket_type = self.stage1_model.predict(X)
        
        # Stage 2: Predict ticket_priority using predicted ticket_type
        predicted_priority = self.stage2_model.predict(X, predicted_ticket_type)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ticket_type': predicted_ticket_type,
            'ticket_priority': predicted_priority
        })
        
        return result
    
    def predict_with_probabilities(self, df):
        """
        Prediction with probability scores.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess features
        text_features = self.text_preprocessor.transform(
            df['ticket_subject'], 
            df['ticket_description']
        )
        structured_features = self.structured_preprocessor.transform(df)
        
        # Combine features
        from preprocessing import FeatureCombiner
        X = FeatureCombiner.combine(text_features, structured_features)
        
        # Stage 1: Predict ticket_type
        predicted_ticket_type = self.stage1_model.predict(X)
        ticket_type_proba = self.stage1_model.predict_proba(X)
        
        # Stage 2: Predict ticket_priority
        predicted_priority = self.stage2_model.predict(X, predicted_ticket_type)
        priority_proba = self.stage2_model.predict_proba(X, predicted_ticket_type)
        
        return {
            'ticket_type': predicted_ticket_type,
            'ticket_type_proba': ticket_type_proba,
            'ticket_type_classes': self.stage1_model.classes_,
            'ticket_priority': predicted_priority,
            'ticket_priority_proba': priority_proba,
            'ticket_priority_classes': self.stage2_model.classes_
        }
    
    def save(self, directory):
        """
        Save complete pipeline to directory.
        
        Args:
            directory: Directory path to save models and preprocessors
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.stage1_model.save(os.path.join(directory, 'stage1_model.pkl'))
        self.stage2_model.save(os.path.join(directory, 'stage2_model.pkl'))
        
        preprocessors = {
            'text': self.text_preprocessor,
            'structured': self.structured_preprocessor
        }
        from preprocessing import save_preprocessors
        save_preprocessors(preprocessors, os.path.join(directory, 'preprocessors.pkl'))
    
    @staticmethod
    def load(directory):
        """
        Load complete pipeline from directory.
        
        Args:
            directory: Directory containing saved models
            
        Returns:
            TwoStagePredictor instance
        """
        import os
        
        stage1_model = Stage1Classifier.load(os.path.join(directory, 'stage1_model.pkl'))
        stage2_model = Stage2Classifier.load(os.path.join(directory, 'stage2_model.pkl'))
        
        from preprocessing import load_preprocessors
        preprocessors = load_preprocessors(os.path.join(directory, 'preprocessors.pkl'))
        
        return TwoStagePredictor(
            stage1_model,
            stage2_model,
            preprocessors['text'],
            preprocessors['structured']
        )
