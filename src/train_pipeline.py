"""
Command-line script for training the complete two-stage pipeline.

Usage:
    python train_pipeline.py --data data/customer_support_tickets.csv --output models/
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_data, split_data, get_default_config, save_config
from training import train_full_pipeline
from evaluation import create_evaluation_report


def main():
    parser = argparse.ArgumentParser(
        description='Train two-stage hierarchical ML pipeline for ticket analysis'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to customer support tickets CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (optional)'
    )
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.7,
        help='Training set proportion (default: 0.7)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Test set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation on test set after training'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("TWO-STAGE HIERARCHICAL ML PIPELINE TRAINING")
    print("=" * 70)
    
    # Load configuration
    if args.config:
        from utils import load_config
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("\nUsing default configuration")
        config = get_default_config()
        config['data']['train_size'] = args.train_size
        config['data']['val_size'] = args.val_size
        config['data']['test_size'] = args.test_size
        config['data']['random_state'] = args.random_state
    
    # Load data
    print(f"\nLoading data from: {args.data}")
    df = load_data(args.data)
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {list(df.columns)}")
    
    # Verify required columns exist
    required_cols = [
        'ticket_subject', 'ticket_description', 'ticket_type', 'ticket_priority',
        'product_purchased', 'ticket_channel', 'customer_gender', 'customer_age'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Split data
    print("\nSplitting data...")
    df_train, df_val, df_test = split_data(
        df,
        train_size=config['data']['train_size'],
        val_size=config['data']['val_size'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    print(f"  Training set: {len(df_train)} samples")
    print(f"  Validation set: {len(df_val)} samples")
    print(f"  Test set: {len(df_test)} samples")
    
    # Train pipeline
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    
    predictor, metrics = train_full_pipeline(df_train, df_val, config)
    
    # Save models
    print(f"\nSaving models to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    predictor.save(args.output)
    
    # Save configuration
    config_path = os.path.join(args.output, 'config.yaml')
    save_config(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Evaluate on test set
    if args.evaluate:
        print("\n" + "=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70)
        
        report_dir = os.path.join(args.output, 'evaluation_report')
        test_results = create_evaluation_report(predictor, df_test, report_dir)
        
        print(f"\n✓ Evaluation complete! Report saved to: {report_dir}")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModels saved to: {args.output}")
    print(f"\nFinal Metrics (Validation Set):")
    print(f"  Stage 1 (Ticket Type) - Macro F1: {metrics['stage1']['macro_f1']:.4f}")
    print(f"  Stage 2 (Ticket Priority) - Macro F1: {metrics['stage2']['macro_f1']:.4f}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
