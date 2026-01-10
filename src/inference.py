"""
Inference module for production predictions.

Provides functions for single and batch predictions using the trained pipeline.
"""

import pandas as pd
import numpy as np
from models import TwoStagePredictor


def predict_single(ticket_data, predictor):
    """
    Make prediction for a single ticket.
    
    Args:
        ticket_data: Dictionary with ticket information:
            - ticket_subject: str
            - ticket_description: str
            - product_purchased: str
            - ticket_channel: str
            - customer_gender: str
            - customer_age: int
        predictor: TwoStagePredictor instance
        
    Returns:
        Dictionary with predictions and probabilities
    """
    # Convert to DataFrame
    df = pd.DataFrame([ticket_data])
    
    # Get predictions with probabilities
    result = predictor.predict_with_probabilities(df)
    
    # Format output
    output = {
        'ticket_type': result['ticket_type'][0],
        'ticket_type_confidence': float(np.max(result['ticket_type_proba'][0])),
        'ticket_priority': result['ticket_priority'][0],
        'ticket_priority_confidence': float(np.max(result['ticket_priority_proba'][0])),
        'ticket_type_probabilities': {
            label: float(prob)
            for label, prob in zip(result['ticket_type_classes'], 
                                  result['ticket_type_proba'][0])
        },
        'ticket_priority_probabilities': {
            label: float(prob)
            for label, prob in zip(result['ticket_priority_classes'],
                                  result['ticket_priority_proba'][0])
        }
    }
    
    return output


def predict_batch(df, predictor):
    """
    Make predictions for a batch of tickets.
    
    Args:
        df: DataFrame with ticket information
        predictor: TwoStagePredictor instance
        
    Returns:
        DataFrame with predictions
    """
    # Get predictions
    predictions = predictor.predict(df)
    
    # Add predictions to original DataFrame
    result_df = df.copy()
    result_df['predicted_ticket_type'] = predictions['ticket_type']
    result_df['predicted_ticket_priority'] = predictions['ticket_priority']
    
    return result_df


def load_pipeline(model_dir):
    """
    Load trained pipeline from directory.
    
    Args:
        model_dir: Directory containing saved models and preprocessors
        
    Returns:
        TwoStagePredictor instance
    """
    return TwoStagePredictor.load(model_dir)


def predict_from_csv(input_csv, output_csv, model_dir):
    """
    Make predictions for tickets in CSV file and save results.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save predictions
        model_dir: Directory containing trained models
    """
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    print(f"Loading model from: {model_dir}")
    predictor = load_pipeline(model_dir)
    
    print("Making predictions...")
    result_df = predict_batch(df, predictor)
    
    print(f"Saving predictions to: {output_csv}")
    result_df.to_csv(output_csv, index=False)
    
    print(f"✓ Predictions complete! Processed {len(result_df)} tickets.")


def interactive_prediction(model_dir):
    """
    Interactive prediction mode for testing.
    
    Args:
        model_dir: Directory containing trained models
    """
    print("Loading model...")
    predictor = load_pipeline(model_dir)
    print("✓ Model loaded successfully!\n")
    
    print("=" * 60)
    print("Interactive Prediction Mode")
    print("=" * 60)
    print("Enter ticket information (or 'quit' to exit)\n")
    
    while True:
        try:
            # Get input
            print("\n" + "-" * 60)
            subject = input("Ticket Subject: ").strip()
            if subject.lower() == 'quit':
                break
            
            description = input("Ticket Description: ").strip()
            product = input("Product Purchased: ").strip()
            channel = input("Ticket Channel (email/chat/phone/social_media): ").strip()
            gender = input("Customer Gender (male/female/other): ").strip()
            age = input("Customer Age: ").strip()
            
            # Create ticket data
            ticket_data = {
                'ticket_subject': subject,
                'ticket_description': description,
                'product_purchased': product.lower().replace(' ', '_'),
                'ticket_channel': channel.lower().replace(' ', '_'),
                'customer_gender': gender.lower(),
                'customer_age': int(age)
            }
            
            # Make prediction
            result = predict_single(ticket_data, predictor)
            
            # Display results
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(f"Ticket Type: {result['ticket_type']}")
            print(f"  Confidence: {result['ticket_type_confidence']:.2%}")
            print(f"\nTicket Priority: {result['ticket_priority']}")
            print(f"  Confidence: {result['ticket_priority_confidence']:.2%}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")
    
    print("\nGoodbye!")
