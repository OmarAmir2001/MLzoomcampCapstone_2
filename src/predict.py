"""
Command-line script for making predictions using trained pipeline.

Usage:
    python predict.py --input data/new_tickets.csv --output predictions.csv --model models/
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from inference import predict_from_csv, interactive_prediction


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained two-stage pipeline'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Directory containing trained models'
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Batch prediction mode
    batch_parser = subparsers.add_parser('batch', help='Batch prediction from CSV')
    batch_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    batch_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save predictions CSV'
    )
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Interactive prediction mode')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        print("\n" + "=" * 60)
        print("BATCH PREDICTION MODE")
        print("=" * 60)
        predict_from_csv(args.input, args.output, args.model)
        
    elif args.mode == 'interactive':
        interactive_prediction(args.model)
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
