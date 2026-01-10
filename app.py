"""
Streamlit Web Application for Two-Stage Ticket Classification

This app provides an interactive interface for predicting ticket types and priorities
using the trained two-stage hierarchical ML model.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import load_pipeline, predict_single, predict_batch


# Page configuration
st.set_page_config(
    page_title="Ticket Classification System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_dir='models/'):
    """Load the trained model pipeline (cached)"""
    try:
        predictor = load_pipeline(model_dir)
        return predictor, None
    except Exception as e:
        return None, str(e)


def main():
    # Header
    st.markdown('<div class="main-header">🎫 Customer Support Ticket Classification</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Two-Stage Hierarchical ML System</div>', 
                unsafe_allow_html=True)
    
    # Load model
    predictor, error = load_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please ensure the model is trained and saved in the 'models/' directory.")
        st.code("python src/train_pipeline.py --data data/customer_support_tickets.csv --output models/")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This application uses a **two-stage hierarchical ML system** to predict:
        
        1. **Ticket Type** (Stage 1)
           - Technical Issue
           - Billing Inquiry
           - Product Inquiry
           - Refund Request
           - Cancellation Request
        
        2. **Ticket Priority** (Stage 2)
           - Critical
           - High
           - Medium
           - Low
        
        ---
        
        ### 🔍 How it works:
        1. Enter ticket details
        2. Stage 1 predicts ticket type
        3. Stage 2 uses predicted type to determine priority
        """)
        
        st.markdown("---")
        st.markdown("**Model Features:**")
        st.markdown("- TF-IDF text features")
        st.markdown("- Structured features")
        st.markdown("- Class imbalance handling")
        st.markdown("- No data leakage")
    
    # Main content - Tabs
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Ticket Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Ticket Information")
            
            ticket_subject = st.text_input(
                "Ticket Subject",
                placeholder="e.g., Product setup issue",
                help="Brief description of the issue"
            )
            
            ticket_description = st.text_area(
                "Ticket Description",
                placeholder="e.g., I'm having trouble setting up my new device...",
                help="Detailed description of the issue",
                height=150
            )
        
        with col2:
            st.subheader("👤 Customer Information")
            
            product_purchased = st.selectbox(
                "Product Purchased",
                options=[
                    "gopro_hero", "lg_smart_tv", "dell_xps", "microsoft_office",
                    "autodesk_autocad", "hp_laptop", "sony_headphones", "apple_iphone",
                    "samsung_galaxy", "canon_eos", "nikon_d3500", "logitech_mouse",
                    "bose_speaker", "fitbit_charge", "amazon_echo"
                ],
                help="Select the product"
            )
            
            ticket_channel = st.selectbox(
                "Ticket Channel",
                options=["email", "chat", "phone", "social_media"],
                help="How the ticket was submitted"
            )
            
            customer_gender = st.selectbox(
                "Customer Gender",
                options=["male", "female", "other"],
                help="Customer's gender"
            )
            
            customer_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=35,
                help="Customer's age"
            )
        
        # Predict button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            predict_button = st.button("🚀 Predict", use_container_width=True, type="primary")
        
        # Make prediction
        if predict_button:
            if not ticket_subject or not ticket_description:
                st.warning("⚠️ Please fill in both ticket subject and description.")
            else:
                with st.spinner("🔮 Predicting..."):
                    # Prepare ticket data
                    ticket_data = {
                        'ticket_subject': ticket_subject,
                        'ticket_description': ticket_description,
                        'product_purchased': product_purchased,
                        'ticket_channel': ticket_channel,
                        'customer_gender': customer_gender,
                        'customer_age': customer_age
                    }
                    
                    # Make prediction
                    result = predict_single(ticket_data, predictor)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Create two columns for results
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="📋 Ticket Type",
                            value=result['ticket_type'].replace('_', ' ').title(),
                            delta=f"{result['ticket_type_confidence']:.1%} confidence"
                        )
                        
                        # Show probabilities
                        with st.expander("View all type probabilities"):
                            probs_df = pd.DataFrame([
                                {"Type": k.replace('_', ' ').title(), "Probability": f"{v:.2%}"}
                                for k, v in result['ticket_type_probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            st.dataframe(probs_df, hide_index=True, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with res_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            label="⚡ Ticket Priority",
                            value=result['ticket_priority'].replace('_', ' ').title(),
                            delta=f"{result['ticket_priority_confidence']:.1%} confidence"
                        )
                        
                        # Show probabilities
                        with st.expander("View all priority probabilities"):
                            probs_df = pd.DataFrame([
                                {"Priority": k.replace('_', ' ').title(), "Probability": f"{v:.2%}"}
                                for k, v in result['ticket_priority_probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            st.dataframe(probs_df, hide_index=True, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Ticket Prediction")
        st.markdown("Upload a CSV file with multiple tickets for batch prediction.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain: ticket_subject, ticket_description, product_purchased, ticket_channel, customer_gender, customer_age"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Clean column names
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                
                # Show preview
                st.subheader("📄 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Predict button
                if st.button("🚀 Predict All", type="primary"):
                    with st.spinner(f"🔮 Predicting {len(df)} tickets..."):
                        # Clean categorical columns
                        categorical_columns = df.select_dtypes(include=['object']).columns
                        for col in categorical_columns:
                            df[col] = df[col].str.lower().str.replace(' ', '_')
                        
                        # Make predictions
                        result_df = predict_batch(df, predictor)
                        
                        # Display results
                        st.success(f"✅ Successfully predicted {len(result_df)} tickets!")
                        
                        st.subheader("📊 Prediction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="ticket_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Show statistics
                        st.subheader("📈 Prediction Statistics")
                        
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            st.markdown("**Ticket Type Distribution:**")
                            type_counts = result_df['predicted_ticket_type'].value_counts()
                            st.bar_chart(type_counts)
                        
                        with stat_col2:
                            st.markdown("**Ticket Priority Distribution:**")
                            priority_counts = result_df['predicted_ticket_priority'].value_counts()
                            st.bar_chart(priority_counts)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Please ensure your CSV has the required columns.")
        
        else:
            # Show example format
            st.info("📝 **Required CSV Format:**")
            example_df = pd.DataFrame({
                'ticket_subject': ['Product setup', 'Billing question'],
                'ticket_description': ['Having trouble...', 'Question about...'],
                'product_purchased': ['gopro_hero', 'lg_smart_tv'],
                'ticket_channel': ['email', 'chat'],
                'customer_gender': ['male', 'female'],
                'customer_age': [35, 42]
            })
            st.dataframe(example_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit | Two-Stage Hierarchical ML System"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
