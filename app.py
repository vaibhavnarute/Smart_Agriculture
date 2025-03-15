import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="AgroBloom-AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import os
import numpy as np
import pandas as pd
import requests
import json
import cv2
from PIL import Image
from io import BytesIO
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from dotenv import load_dotenv
import google.generativeai as genai
from disease import analyze_plant_disease
from PIL import UnidentifiedImageError
import ee
import geemap
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta

# Initialize Earth Engine with error handling
def initialize_ee():
    try:
        # Try to initialize with default credentials
        project_id = os.getenv("GEE_PROJECT_ID", "agrobloom-ai")  # Get from environment or use default
        ee.Initialize(project=project_id)
    except Exception as e:
        try:
            # If failed, try to authenticate and initialize
            ee.Authenticate()
            project_id = os.getenv("GEE_PROJECT_ID", "agrobloom-ai")
            ee.Initialize(project=project_id)
        except Exception as auth_e:
            st.error(f"""
                Error initializing Google Earth Engine. Please follow these steps:
                1. Run 'earthengine authenticate' in your terminal
                2. Create a Google Cloud Project and enable Earth Engine
                3. Set up credentials for Earth Engine
                4. Add your project ID to the .env file as GEE_PROJECT_ID
                
                Error details: {str(auth_e)}
            """)
            return False
    return True

# Initialize Earth Engine
ee_initialized = initialize_ee()

# Function to get soil moisture data from Google Earth Engine
def get_soil_moisture(lat, lon):
    if not ee_initialized:
        st.error("Earth Engine not initialized. Cannot fetch soil moisture data.")
        return None
        
    try:
        # Create a point and buffer it to create a small region
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(1000)  # 1km buffer
        
        # Get the current date and the date 7 days ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Load the ERA5-Land hourly data
        collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')\
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))\
            .select('volumetric_soil_water_layer_1')  # This is the correct band name for soil moisture
            
        # Check if we have any images
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            st.warning("No recent soil moisture data available. Extending search to last 30 days.")
            start_date = end_date - timedelta(days=30)
            collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')\
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))\
                .select('volumetric_soil_water_layer_1')
            collection_size = collection.size().getInfo()
            
            if collection_size == 0:
                st.error("No soil moisture data available for this location in the last 30 days.")
                return None
        
        try:
            # Get the mean value over the region for each image
            def get_region_mean(image):
                mean = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=1000
                )
                return image.set('mean', mean.get('volumetric_soil_water_layer_1'))

            # Map over the collection and get mean values
            collection_with_means = collection.map(get_region_mean)
            
            # Get the most recent non-null value
            sorted_collection = collection_with_means.sort('system:time_start', False)
            recent_value = sorted_collection.first().get('mean').getInfo()
            
            if recent_value is None:
                st.error("No valid soil moisture data available for this location.")
                return None
                
            # Get the date range for reference
            st.info(f"Soil moisture data from: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
            # ERA5-Land volumetric soil water is in m³/m³, convert to percentage
            return round(recent_value * 100, 2)
            
        except Exception as sample_error:
            st.error(f"Error processing soil moisture data: {str(sample_error)}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching soil moisture data: {str(e)}")
        return None

# Function to display soil moisture map
def display_soil_moisture_map(lat, lon, soil_moisture):
    try:
        # Create a map centered at the location
        m = folium.Map(location=[lat, lon], zoom_start=10)
        
        # Add a marker with soil moisture information
        folium.Marker(
            [lat, lon],
            popup=f"Soil Moisture: {soil_moisture}%",
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)
        
        # Add a circle to represent the area of measurement
        folium.Circle(
            location=[lat, lon],
            radius=2000,  # 2km radius
            color='blue',
            fill=True,
            popup='Measurement Area'
        ).add_to(m)
        
        # Display the map in Streamlit
        folium_static(m)
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")

# Function to get coordinates from city name
def get_coordinates(city):
    try:
        # Using OpenStreetMap Nominatim API for geocoding
        url = f"https://nominatim.openstreetmap.org/search?city={city}&format=json"
        headers = {
            'User-Agent': 'AgroBloom-AI/1.0'  # Add user agent to comply with Nominatim usage policy
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        return None, None
    except Exception as e:
        st.error(f"Error getting coordinates: {str(e)}")
        return None, None

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load CSS
def local_css(file_name):
    try:
        with open(os.path.join(os.path.dirname(__file__), file_name)) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS file {file_name}: {e}")

local_css("style.css")

# -------------------- Load Lottie Animations --------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_agri = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ygiuluqn.json")

# -------------------- FUNCTIONS --------------------

# Function to get weather data from API
def get_weather_data(city):
    api_key = os.getenv("API_KEY")
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    return data

# Function to load CSV data
def load_data():
    try:
        soil_data = pd.read_csv("soil_analysis_data.csv")
        crop_production_data = pd.read_csv("crop_production_data.csv")
        return soil_data, crop_production_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to analyze soil health
def analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter):
    healthy = {'pH': (6.0, 7.5), 'nitrogen': (20, 50), 'phosphorus': (15, 40), 'potassium': (15, 40), 'organic_matter': (3, 6)}
    moderate = {'pH': (5.5, 6.0), 'nitrogen': (10, 20), 'phosphorus': (10, 15), 'potassium': (10, 15), 'organic_matter': (2, 3)}

    pH_status = 'Healthy' if healthy['pH'][0] <= pH <= healthy['pH'][1] else ('Moderate' if moderate['pH'][0] <= pH <= moderate['pH'][1] else 'Unhealthy')
    nitrogen_status = 'Healthy' if healthy['nitrogen'][0] <= nitrogen <= healthy['nitrogen'][1] else ('Moderate' if moderate['nitrogen'][0] <= nitrogen <= moderate['nitrogen'][1] else 'Unhealthy')
    phosphorus_status = 'Healthy' if healthy['phosphorus'][0] <= phosphorus <= healthy['phosphorus'][1] else ('Moderate' if moderate['phosphorus'][0] <= phosphorus <= moderate['phosphorus'][1] else 'Unhealthy')
    potassium_status = 'Healthy' if healthy['potassium'][0] <= potassium <= healthy['potassium'][1] else ('Moderate' if moderate['potassium'][0] <= potassium <= moderate['potassium'][1] else 'Unhealthy')
    organic_matter_status = 'Healthy' if healthy['organic_matter'][0] <= organic_matter <= healthy['organic_matter'][1] else ('Moderate' if moderate['organic_matter'][0] <= organic_matter <= moderate['organic_matter'][1] else 'Unhealthy')

    overall_health = {
        'pH': pH_status,
        'Nitrogen': nitrogen_status,
        'Phosphorus': phosphorus_status,
        'Potassium': potassium_status,
        'Organic Matter': organic_matter_status
    }

    return overall_health

# Function to process satellite images (example placeholder)
def process_satellite_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to train the crop recommendation model
def train_crop_recommendation_model(soil_data, crop_production_data):
    merged_data = pd.merge(soil_data, crop_production_data, on='District')
    features = merged_data[['pH Level', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)', 'Organic Matter (%)']]
    target = merged_data['Crop']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Crop recommendation model accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, "crop_recommendation_model.pkl")
    st.write("Crop recommendation model trained and saved.")
    return model

# Function to load the trained crop recommendation model
def load_crop_recommendation_model():
    try:
        model = joblib.load("crop_recommendation_model.pkl")
        st.write("Crop recommendation model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No trained crop recommendation model found. Please train a new model first.")
        return None

# Function to recommend crops based on soil data using the trained model
def recommend_crops_with_model(model, soil_data_row):
    prediction = model.predict([soil_data_row])
    return prediction[0]

# Function to get historical weather data for training the irrigation model (example data)
def get_historical_weather_data():
    return pd.DataFrame({
        'temperature': [22, 24, 20, 23, 25],
        'humidity': [60, 65, 70, 55, 50],
        'precipitation': [5, 0, 10, 0, 0],
        'soil_moisture': [30, 28, 35, 33, 30]
    })

# Function to train the irrigation model
def train_irrigation_model():
    data = get_historical_weather_data()
    X = data[['temperature', 'humidity', 'precipitation']]
    y = data['soil_moisture']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "irrigation_model.pkl")
    st.write("Irrigation model trained and saved.")
    return model

# Function to load the irrigation model
def load_irrigation_model():
    try:
        model = joblib.load("irrigation_model.pkl")
        st.write("Irrigation model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No trained irrigation model found. Please train a new model first.")
        return None

# Function for irrigation management with predictive analytics
def irrigation_management(weather_data, soil_moisture):
    model = load_irrigation_model()
    
    if model:
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        precipitation = weather_data.get('rain', {}).get('1h', 0)

        prediction = model.predict([[temp, humidity, precipitation]])
        predicted_soil_moisture = prediction[0]

        st.write(f"Current Soil Moisture: {soil_moisture}%")
        st.write(f"Predicted Soil Moisture: {predicted_soil_moisture:.2f}%")

        if soil_moisture < predicted_soil_moisture:
            st.warning("Irrigation needed to reach optimal soil moisture levels.")
        else:
            st.success("Soil moisture is sufficient; no additional irrigation required.")
    else:
        st.error("Unable to perform irrigation management without a trained model.")

# -------------------- Main App --------------------

def main():
    st.sidebar.title("🌾 AgroBloom AI")
    menu = [
        "Home", 
        "Weather Forecasting", 
        "Irrigation Management", 
        "Soil Health Analysis", 
        "Crop Recommendation",
        "Disease Detection"
    ]
    choice = st.sidebar.radio("Navigation", menu)

    # -------------------- Home Page --------------------
    if choice == "Home":
        st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 AgroBloom AI</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Hero Section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                """
                <div style='text-align: justify; font-size: 20px;'>
                    Welcome to <strong>AgroBloom AI</strong> - Your intelligent farming companion! 
                    Harness the power of artificial intelligence to optimize your agricultural operations, 
                    increase crop yield, and make data-driven decisions for sustainable farming.
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
            features = [
                {"icon": "🌤️", "title": "Smart Weather Insights", "desc": "Real-time weather predictions and adaptive planning"},
                {"icon": "💧", "title": "AI Irrigation System", "desc": "Optimized water usage with predictive analytics"},
                {"icon": "📊", "title": "Soil Health Dashboard", "desc": "Comprehensive nutrient analysis and recommendations"},
                {"icon": "🌿", "title": "Disease Detection", "desc": "AI-powered plant disease detection and treatment recommendations"}
            ]
            
            for feat in features:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <span class="feature-icon">{feat['icon']}</span>
                        <h3 class="feature-title">{feat['title']}</h3>
                        <p class="feature-desc">{feat['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        
        with col2:
            st_lottie(lottie_agri, height=400, key="agri")
        
        st.markdown("---")
        
        # Stats Section
        st.subheader("🚀 Get Started")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Farmers Served", "1.2K+", "34 New")
        with cols[1]:
            st.metric("Crop Accuracy", "92%", "5% Increase")
        with cols[2]:
            st.metric("Water Saved", "4.7M L", "12% Efficiency")
        with cols[3]:
            st.metric("Yield Improved", "65%", "8% MoM")
        style_metric_cards()
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; padding: 20px;'>
                <h3>🌍 Join the Smart Farming Revolution</h3>
                <p>Start your journey towards sustainable and efficient agriculture today!</p>
            </div>
            """, unsafe_allow_html=True
        )

    # -------------------- Weather Forecasting --------------------
    elif choice == "Weather Forecasting":
        colored_header(
            label="🌤️ Smart Weather Insights",
            description="Real-time weather predictions and farming recommendations",
            color_name="green-70"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            city = st.text_input("📍 Enter Location", "London")
            if st.button("Get Weather Analysis", use_container_width=True):
                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data(city)
                    if weather_data and weather_data.get("main"):
                        with col2:
                            st.subheader(f"Weather Report for {city}")
                            cols = st.columns(4)
                            cols[0].metric("Temperature", f"{weather_data['main']['temp']}°C",
                                           help="Optimal range for most crops: 15-30°C")
                            cols[1].metric("Humidity", f"{weather_data['main']['humidity']}%",
                                           "Ideal range: 40-80%")
                            cols[2].metric("Precipitation",
                                           f"{weather_data.get('rain', {}).get('1h', 0)}mm",
                                           "Next 3 hours")
                            cols[3].metric("Wind Speed",
                                           f"{weather_data['wind']['speed']} m/s",
                                           "Wind direction")
                            style_metric_cards()
                            
                            # Simple weather advisory based on temperature
                            temp = weather_data['main']['temp']
                            if temp < 10:
                                advisory = "❄️ Frost alert! Protect sensitive crops."
                            elif 10 <= temp < 20:
                                advisory = "🌤️ Cool weather - ideal for leafy greens."
                            elif 20 <= temp < 30:
                                advisory = "🌞 Optimal growing conditions."
                            else:
                                advisory = "🔥 Heat stress warning! Increase irrigation."
                            
                            st.markdown(
                                f"""
                                <div class="advisory-box">
                                    <h3>🌱 Farming Advisory</h3>
                                    <p>{advisory}</p>
                                </div>
                                """, unsafe_allow_html=True
                            )
                    else:
                        st.error("Failed to retrieve weather data. Please check the city name or API key.")

    # -------------------- Irrigation Management --------------------
    elif choice == "Irrigation Management":
        colored_header(
            label="💧 Smart Irrigation System",
            description="AI-powered water management with real-time soil moisture data",
            color_name="blue-70"
        )
        
        # Define suggested agricultural regions
        suggested_regions = {
            "India": [
                "Punjab, India",
                "Haryana, India",
                "Uttar Pradesh, India",
                "Bihar, India",
                "West Bengal, India",
                "Maharashtra, India",
                "Karnataka, India",
                "Tamil Nadu, India",
                "Andhra Pradesh, India",
                "Madhya Pradesh, India",
                "Gujarat, India",
                "Rajasthan, India"
            ],
            "United States": [
                "Sacramento, California",
                "Fresno, California",
                "Des Moines, Iowa"
            ],
            "Europe": [
                "Bordeaux, France",
                "Tuscany, Italy",
                "Andalusia, Spain"
            ]
        }
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Field Parameters")
            
            # Region selection
            region_category = st.selectbox("Select Region Category", ["India", "United States", "Europe"])
            city = st.selectbox("📍 Select Location", suggested_regions[region_category])
            
            # Alternative manual input
            use_custom_location = st.checkbox("Use Custom Location")
            if use_custom_location:
                city = st.text_input("Enter Custom Location")
            
            crop_type = st.selectbox("Crop Type", ["Wheat", "Corn", "Rice", "Soybean"])
            st.caption("💡 Optimal moisture levels vary by crop type")
            
            # Add region information
            if not use_custom_location:
                if region_category == "India":
                    st.info("""
                    🌾 Major Crops in Selected Region:
                    - Punjab: Wheat, Rice, Cotton
                    - Haryana: Wheat, Rice, Sugarcane
                    - UP: Wheat, Rice, Sugarcane
                    - Bihar: Rice, Wheat, Maize
                    - West Bengal: Rice, Jute, Tea
                    - Maharashtra: Cotton, Sugarcane, Soybean
                    - Karnataka: Coffee, Sugarcane, Ragi
                    - Tamil Nadu: Rice, Sugarcane, Cotton
                    - Andhra Pradesh: Rice, Cotton, Sugarcane
                    - Madhya Pradesh: Soybean, Wheat, Rice
                    - Gujarat: Cotton, Groundnut, Wheat
                    - Rajasthan: Wheat, Millet, Pulses
                    """)
            
            if st.button("Calculate Irrigation", use_container_width=True):
                with st.spinner("Analyzing field conditions..."):
                    # Get coordinates from city name
                    lat, lon = get_coordinates(city)
                    
                    if lat and lon:
                        # Get real-time soil moisture data
                        soil_moisture = get_soil_moisture(lat, lon)
                        
                        if soil_moisture is not None:
                            weather_data = get_weather_data(city)
                            if weather_data and weather_data.get("main"):
                                with col2:
                                    st.subheader("Irrigation Plan")
                                    
                                    # Display the soil moisture map
                                    st.write("### 🗺️ Soil Moisture Map")
                                    display_soil_moisture_map(lat, lon, soil_moisture)
                                    
                                    # Display current conditions
                                    st.write("### 📊 Current Conditions")
                                    cols = st.columns(3)
                                    cols[0].metric(
                                        "Soil Moisture",
                                        f"{soil_moisture}%",
                                        delta="Real-time data",
                                        help="Real-time soil moisture from satellite data"
                                    )
                                    cols[1].metric(
                                        "Temperature",
                                        f"{weather_data['main']['temp']}°C",
                                        help="Current temperature"
                                    )
                                    cols[2].metric(
                                        "Humidity",
                                        f"{weather_data['main']['humidity']}%",
                                        help="Current humidity"
                                    )
                                    
                                    # Run irrigation management with real-time data
                                    st.write("### 💧 Irrigation Analysis")
                                    irrigation_management(weather_data, soil_moisture)
                                    
                                    # Visual moisture indicator with improved styling
                                    st.markdown(
                                        f"""
                                        <div style="margin: 20px 0;">
                                            <h4>Soil Moisture Level</h4>
                                            <div class="moisture-gauge" style="
                                                background: #f0f2f6;
                                                border-radius: 10px;
                                                height: 30px;
                                                width: 100%;
                                                overflow: hidden;
                                            ">
                                                <div style="
                                                    background: linear-gradient(90deg, #2E8B57, #3CB371);
                                                    width: {soil_moisture}%;
                                                    height: 100%;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    color: white;
                                                    transition: width 0.5s ease-in-out;
                                                ">
                                                    {soil_moisture}%
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                                    
                                    # Crop-specific recommendations
                                    optimal_moisture = {
                                        "Wheat": (30, 50),
                                        "Corn": (35, 55),
                                        "Rice": (60, 85),
                                        "Soybean": (40, 60)
                                    }
                                    
                                    crop_range = optimal_moisture[crop_type]
                                    st.write(f"### 🌾 Crop-Specific Analysis for {crop_type}")
                                    st.write(f"Optimal soil moisture range: {crop_range[0]}% - {crop_range[1]}%")
                                    
                                    if soil_moisture < crop_range[0]:
                                        st.warning(f"⚠️ Soil moisture is below optimal range for {crop_type}. Irrigation recommended.")
                                    elif soil_moisture > crop_range[1]:
                                        st.warning(f"⚠️ Soil moisture is above optimal range for {crop_type}. Reduce irrigation.")
                                    else:
                                        st.success(f"✅ Soil moisture is within optimal range for {crop_type}.")
                                    
                                    # Water conservation stats
                                    st.markdown(
                                        """
                                        <div style="
                                            background: #f8f9fa;
                                            border-radius: 10px;
                                            padding: 20px;
                                            margin-top: 20px;
                                        ">
                                            <h3>💧 Water Conservation Impact</h3>
                                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                                <div>
                                                    <h4>Monthly Savings</h4>
                                                    <p>Water saved: <strong>12,500L</strong></p>
                                                    <p>Cost reduction: <strong>15%</strong></p>
                                                </div>
                                                <div>
                                                    <h4>Environmental Impact</h4>
                                                    <p>Carbon footprint reduction: <strong>25%</strong></p>
                                                    <p>Sustainability score: <strong>8.5/10</strong></p>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                            else:
                                st.error("Failed to retrieve weather data for irrigation management.")
                        else:
                            st.error("Failed to retrieve soil moisture data. Please try again.")
                    else:
                        st.error("Could not find coordinates for the specified location. Please check the city name.")

    # -------------------- Soil Health Analysis --------------------
    elif choice == "Soil Health Analysis":
        colored_header(
            label="🌱 Soil Health Dashboard",
            description="Comprehensive soil nutrient analysis and recommendations",
            color_name="orange-70"
        )
        
        with st.expander("🔍 How to use this tool"):
            st.markdown(
                """
                1. Input your soil test results below.<br>
                2. Click 'Analyze Soil Health'.<br>
                3. Receive customized recommendations.
                """, unsafe_allow_html=True
            )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Soil Parameters")
            pH = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1)
            nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=20.0)
            phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=15.0)
            potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, value=15.0)
            organic_matter = st.slider("Organic Matter (%)", 0.0, 100.0, 5.0)
            
            if st.button("Analyze Soil Health", use_container_width=True):
                with st.spinner("Analyzing soil composition..."):
                    health_status = analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        st.markdown(
                            f"""
                            <div class="gauge-container">
                                <p><strong>pH:</strong> {health_status['pH']}</p>
                                <p><strong>Nitrogen:</strong> {health_status['Nitrogen']}</p>
                                <p><strong>Phosphorus:</strong> {health_status['Phosphorus']}</p>
                                <p><strong>Potassium:</strong> {health_status['Potassium']}</p>
                                <p><strong>Organic Matter:</strong> {health_status['Organic Matter']}</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.markdown(
                            """
                            <div class="recommendation-box">
                                <h3>📋 Recommended Actions</h3>
                                <ul>
                                    Apply organic compost as per soil test recommendations.
                                    Ideal Soil pH levels between 6.0 - 7.5.
                                    Ideal Nitrogen levels between 20 - 50
                                    Ideal Phosphorus levels between 15 - 40
                                    Ideal potassium levels between 15 - 40
                                    Retest soil after 45 days.
                                </ul>
                            </div>
                            """, unsafe_allow_html=True
                        )

    # -------------------- Crop Recommendation --------------------
    elif choice == "Crop Recommendation":
        colored_header(
            label="🌾 Smart Crop Advisor",
            description="AI-powered crop recommendations based on soil and climate",
            color_name="violet-70"
        )
        
        soil_data, crop_production_data = load_data()
        
        if soil_data is not None and crop_production_data is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Field Conditions")
                pH = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1)
                nitrogen = st.number_input("Nitrogen Level (kg/ha)", min_value=0.0, value=20.0)
                phosphorus = st.number_input("Phosphorus Level (kg/ha)", min_value=0.0, value=15.0)
                potassium = st.number_input("Potassium Level (kg/ha)", min_value=0.0, value=15.0)
                organic_matter = st.slider("Organic Matter Content (%)", 0.0, 100.0, 5.0)
                
                if st.button("Get Crop Recommendations", use_container_width=True):
                    with st.spinner("Analyzing optimal crops..."):
                        model = load_crop_recommendation_model()
                        if model:
                            soil_data_row = [pH, nitrogen, phosphorus, potassium, organic_matter]
                            recommended_crop = recommend_crops_with_model(model, soil_data_row)
                            
                            with col2:
                                st.markdown(
                                    f"""
                                    <div class="crop-card">
                                        <h2>Recommended Crop</h2>
                                        <h1 class="crop-name">🌽 {recommended_crop}</h1>
                                        <div class="crop-stats">
                                            <div>
                                                <h3>Expected Yield</h3>
                                                <p>12-15 tons/ha</p>
                                            </div>
                                            <div>
                                                <h3>Best Season</h3>
                                                <p>Kharif/Rabi</p>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                        else:
                            st.error("Crop recommendation model not loaded. Please train the model first.")
        else:
            st.error("Data files not loaded. Please ensure CSV files are available.")

    # -------------------- Disease Detection --------------------
    elif choice == "Disease Detection":
        colored_header(
            label="🌿 Disease Detection",
            description="AI-powered plant disease detection and treatment recommendations",
            color_name="red-70"
        )
        
        with st.expander("🔍 How to use this tool"):
            st.markdown(
                """
                1. Upload a plant image.<br>
                2. Click 'Analyze Disease'.<br>
                3. Receive disease detection and treatment recommendations.
                """, unsafe_allow_html=True
            )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png"])
            if uploaded_file:
                try:
                    # Ensure the file is an image
                    image = Image.open(BytesIO(uploaded_file.read()))
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("Analyze Disease", use_container_width=True):
                        with st.spinner("Analyzing disease..."):
                            disease_info = analyze_plant_disease(image)
                            
                            with col2:
                                st.subheader("Disease Detection Results")
                                st.write(disease_info)
                                st.markdown(
                                    """
                                    <div class="recommendation-box">
                                        <h3>📋 Recommended Actions</h3>
                                        <ul>
                                            <li>Consult a local agricultural expert for further diagnosis.</li>
                                            <li>Apply recommended treatment methods.</li>
                                            <li>Monitor plant health regularly.</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                except UnidentifiedImageError:
                    st.error("The uploaded file is not a valid image. Please upload a valid image file.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
