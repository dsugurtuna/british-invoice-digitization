import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RoyalAudit Digitizer", page_icon="🇬🇧", layout="wide")

st.title("🇬🇧 RoyalAudit Digitizer Dashboard")
st.markdown("""
**Client:** UK Digital Audit Solutions Ltd.  
**System:** Automated Invoice Field Extraction  
**Status:** Production Ready
""")

# Sidebar
st.sidebar.header("Configuration")
api_status = "🔴 Offline"
try:
    r = requests.get(f"{API_URL}/health/live", timeout=2)
    if r.status_code == 200:
        api_status = "🟢 Online"
except:
    pass
st.sidebar.write(f"API Status: {api_status}")
st.sidebar.write(f"API URL: `{API_URL}`")

uploaded_file = st.file_uploader("Upload Invoice Scan", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Invoice', use_column_width=True)
    
    if st.button("Analyze Invoice"):
        if api_status == "🔴 Offline":
            st.error("API is offline. Please start the backend service.")
        else:
            with st.spinner("Processing with YOLOv5x..."):
                try:
                    # Prepare file for upload
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    files = {'file': ('invoice.jpg', img_byte_arr, 'image/jpeg')}
                    
                    # Call API
                    response = requests.post(f"{API_URL}/detect", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("data", {})
                        detections = data.get("detections", [])
                        
                        st.success(f"Analysis Complete. Found {len(detections)} fields.")
                        
                        # Display results in a table
                        if detections:
                            df = pd.DataFrame(detections)
                            # Select relevant columns
                            cols = ["class_name", "confidence"]
                            if "text" in df.columns:
                                cols.append("text")
                            
                            st.dataframe(df[cols].style.format({"confidence": "{:.2%}"}))
                            
                            # JSON view
                            with st.expander("View Raw JSON"):
                                st.json(result)
                        else:
                            st.info("No fields detected.")
                            
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

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
