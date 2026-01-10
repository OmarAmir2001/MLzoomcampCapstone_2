"""
Two-Stage Hierarchical ML System for Customer Support Ticket Analysis

This package provides a modular, production-ready implementation of a two-stage
hierarchical machine learning system for predicting ticket types and priorities.
"""

__version__ = '1.0.0'
__author__ = 'ML Zoomcamp Capstone Project'

from .preprocessing import TextPreprocessor, StructuredFeatureEncoder, FeatureCombiner
from .models import Stage1Classifier, Stage2Classifier, TwoStagePredictor
from .training import train_stage1, train_stage2, train_full_pipeline
from .evaluation import evaluate_stage1, evaluate_stage2, evaluate_full_pipeline
from .inference import predict_single, predict_batch, load_pipeline

__all__ = [
    'TextPreprocessor',
    'StructuredFeatureEncoder',
    'FeatureCombiner',
    'Stage1Classifier',
    'Stage2Classifier',
    'TwoStagePredictor',
    'train_stage1',
    'train_stage2',
    'train_full_pipeline',
    'evaluate_stage1',
    'evaluate_stage2',
    'evaluate_full_pipeline',
    'predict_single',
    'predict_batch',
    'load_pipeline'
]
