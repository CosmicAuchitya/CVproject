import streamlit as st
import base64
import os

# पेज सेटअप (Wide Mode ताकि PDF अच्छे से दिखे)
st.set_page_config(page_title="Airline Dashboard", layout="wide")

def show_pdf(file_path):
    # चेक करें कि फाइल है या नहीं
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # PDF दिखाने के लिए HTML
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        
        # स्ट्रीमलिट में रेंडर करें
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"File not found: {file_path}")

# टाइटल
st.title("📊 US Airline Data Dashboard")

# यहाँ अपनी PDF फाइल का नाम लिखें जो रेपो में है
pdf_file_name = "AirlineDashbored.pdf" 

# फंक्शन कॉल
show_pdf(pdf_file_name)