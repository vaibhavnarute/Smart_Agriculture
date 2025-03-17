import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-flash 1.5')

def generate_virtual_soil_data(farm_name, region):
    """
    Generate realistic soil health data using an LLM.
    Returns a dictionary with soil parameters.
    """
    try:
        prompt = f"""
        Generate realistic soil health data for a virtual farm named "{farm_name}" located in {region}.
        Provide the following parameters in JSON format:
        - pH (between 4.0 and 9.0)
        - Nitrogen (kg/ha, between 10 and 100)
        - Phosphorus (kg/ha, between 5 and 50)
        - Potassium (kg/ha, between 10 and 100)
        - Organic Matter (%, between 1 and 10)
        - Soil Moisture (%, between 20 and 80)
        - Soil Type (e.g., Loamy, Sandy, Clayey)
        - Region-Specific Notes (e.g., suitability for crops, common issues)
        """
        response = model.generate_content(prompt)
        try:
            soil_data = json.loads(response.text)
            return soil_data
        except json.JSONDecodeError:
            soil_data = {}
            for line in response.text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    soil_data[key.strip()] = value.strip()
            return soil_data
    except Exception as e:
        st.error(f"Error generating soil data: {str(e)}")
        return None

def generate_random_soil_data():
    """
    Generate random soil data for automated analysis.
    Returns a dictionary with soil parameters.
    """
    soil_data = {
        "pH": round(np.random.uniform(4.0, 9.0), 1),  # pH between 4.0 and 9.0, rounded to 1 decimal place
        "Nitrogen": round(np.random.uniform(10, 100)),  # Nitrogen between 10 and 100 kg/ha
        "Phosphorus": round(np.random.uniform(5, 50)),  # Phosphorus between 5 and 50 kg/ha
        "Potassium": round(np.random.uniform(10, 100)),  # Potassium between 10 and 100 kg/ha
        "Organic Matter": round(np.random.uniform(1, 10), 1),  # Organic Matter between 1 and 10%
        "Soil Moisture": round(np.random.uniform(20, 80)),  # Soil Moisture between 20 and 80%
        "Soil Type": np.random.choice(["Loamy", "Sandy", "Clayey"]),
        "Region-Specific Notes": "Suitable for general crops."
    }
    return soil_data

# ------------------------- Sidebar Navigation -------------------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Select Option", ["Home", "Soil Health Analysis"])

# ------------------------- Main App -------------------------
if choice == "Soil Health Analysis":
    st.header("🌱 Soil Health Dashboard")
    st.write("Comprehensive soil nutrient analysis and recommendations.")

    with st.expander("🔍 How to use this tool"):
        st.markdown("""
        1. Input your soil test results below.<br>
        2. Click 'Analyze Soil Health'.<br>
        3. Receive customized recommendations.
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Virtual Farm Setup")
        
        # Input fields for virtual farm
        farm_name = st.text_input("Enter Virtual Farm Name", "Green Valley Farm")
        region = st.selectbox("Select Region", ["Punjab, India", "California, USA", "Tuscany, Italy"])
        
        if st.button("Generate Virtual Soil Data", use_container_width=True):
            with st.spinner("Generating realistic soil data..."):
                # For this example we use random generation.
                soil_data = generate_random_soil_data()
                if soil_data:
                    st.session_state['virtual_soil_data'] = soil_data
                    st.success("Virtual soil data generated successfully!")
                else:
                    st.error("Failed to generate soil data. Please try again.")
        
        # Display generated soil data
        if 'virtual_soil_data' in st.session_state:
            st.subheader("Generated Soil Data")
            st.json(st.session_state['virtual_soil_data'])
            
            # Auto-populate soil parameters
            pH = float(st.session_state['virtual_soil_data'].get('pH', 7.0))
            nitrogen = float(st.session_state['virtual_soil_data'].get('Nitrogen', 20.0))
            phosphorus = float(st.session_state['virtual_soil_data'].get('Phosphorus', 15.0))
            potassium = float(st.session_state['virtual_soil_data'].get('Potassium', 15.0))
            organic_matter = float(st.session_state['virtual_soil_data'].get('Organic Matter', 5.0))
        else:
            # Default values if no virtual data is generated
            pH = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1)
            nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=20.0)
            phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=15.0)
            potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, value=15.0)
            organic_matter = st.slider("Organic Matter (%)", 0.0, 100.0, 5.0)
    
    if st.button("Analyze Soil Health", use_container_width=True):
        with st.spinner("Analyzing soil composition..."):
            # Simple analysis logic
            health_status = {
                'pH': "Healthy" if 6.0 <= pH <= 7.5 else "Unhealthy",
                'Nitrogen': "Healthy" if 20 <= nitrogen <= 50 else "Unhealthy",
                'Phosphorus': "Healthy" if 15 <= phosphorus <= 40 else "Unhealthy",
                'Potassium': "Healthy" if 15 <= potassium <= 40 else "Unhealthy",
                'Organic Matter': "Healthy" if 3 <= organic_matter <= 6 else "Unhealthy"
            }
            with col2:
                st.subheader("Analysis Results")
                st.json(health_status)
                st.markdown("""
                ### 📋 Recommended Actions
                - Apply organic compost as per soil test recommendations.
                - Maintain soil pH between 6.0 - 7.5.
                - Ensure Nitrogen levels remain between 20 - 50 kg/ha.
                - Keep Phosphorus between 15 - 40 kg/ha.
                - Adjust Potassium to be within 15 - 40 kg/ha.
                - Retest soil after 45 days.
                """)

elif choice == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is a placeholder for home content.")
    st.markdown("---")
    st.write("Please select an option from the sidebar to continue.")