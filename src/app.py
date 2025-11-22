import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path to import inference module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock import if inference.py dependencies are missing in this specific env
try:
    from inference import InvoiceDigitizer
except ImportError:
    st.error("Could not import InvoiceDigitizer. Ensure requirements are installed.")

st.set_page_config(page_title="RoyalAudit Digitizer", page_icon="🇬🇧", layout="wide")

st.title("🇬🇧 RoyalAudit Digitizer Dashboard")
st.markdown("""
**Client:** UK Digital Audit Solutions Ltd.  
**System:** Automated Invoice Field Extraction  
""")

st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
model_path = st.sidebar.text_input("Model Path", "../models/royal_audit_v1_best.pt")

uploaded_file = st.file_uploader("Upload Invoice Scan", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Invoice', use_column_width=True)
    
    if st.button("Analyze Invoice"):
        with st.spinner("Processing with YOLOv5x..."):
            try:
                # Initialize model (caching this in production would be better)
                if not os.path.exists(model_path):
                    st.warning("Model file not found. Using dummy data for demonstration.")
                    # Dummy data for demo purposes if model isn't trained yet
                    detections = [
                        {"label": "Total Amount", "confidence": 0.98, "value": "£1,250.00"},
                        {"label": "Invoice Date", "confidence": 0.95, "value": "12/05/1998"},
                        {"label": "Vendor Name", "confidence": 0.92, "value": "British Telecom"},
                        {"label": "VAT Amount", "confidence": 0.89, "value": "£218.75"}
                    ]
                    st.success("Analysis Complete (Simulation)")
                else:
                    digitizer = InvoiceDigitizer(model_path=model_path, conf_thres=confidence_threshold)
                    # Convert PIL to numpy for the inference class
                    img_np = np.array(image)
                    result = digitizer.process_image(img_np)
                    detections = result['detections']
                    st.success("Analysis Complete")

                # Display Results
                st.subheader("Extracted Fields")
                
                # Create a nice dataframe
                data = []
                for d in detections:
                    # If we had OCR, we would have the 'value' here. 
                    # Since this is just YOLO, we show the label and confidence.
                    data.append({
                        "Field": d['label'],
                        "Confidence": f"{d['confidence']:.2%}",
                        "Location": str(d.get('bbox', 'N/A'))
                    })
                
                df = pd.DataFrame(data)
                st.table(df)
                
                # JSON Export
                st.json(detections)

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.info("Developed for UK Digital Audit Solutions Ltd. - Confidential")
