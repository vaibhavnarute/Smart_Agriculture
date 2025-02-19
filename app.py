import os
import numpy as np
import pandas as pd
import requests
import json
import cv2
from PIL import Image
from io import BytesIO
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AgroBloom-AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
def local_css(file_name):
    try:
        with open(file_name) as f:
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
    api_key = "f853c70d8e3e9c8c1bf23747806c73a6"
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

# Function to create a simple CNN model
# Create CNN Model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout to prevent overfitting
        Dense(1, activation='sigmoid')  # Binary classification (Healthy = 0, Unhealthy = 1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Train Model on Placeholder Data
def train_placeholder_model():
    model = create_cnn_model()
    
    X_train = np.random.rand(100, 64, 64, 3)  # 100 samples of 64x64 images
    y_train = np.random.randint(2, size=100)   # Binary labels (0=Healthy, 1=Unhealthy)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Class weights to handle potential class imbalance
    class_weights = {0: 1, 1: 10}  # Adjust weights based on class imbalance

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])
    model.save('crop_health_model.h5')
    
    st.write("Crop health model trained and saved.")
    return model

# Preprocess Image Before Prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error("Error loading image. Please check the file path.")
        return None
    
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image

# Predict Crop Health
def predict_crop_health(model, image_path):
    image = preprocess_image(image_path)
    if image is not None:
        prediction = model.predict(image)[0][0]  # Get scalar output
        print(f"Prediction probability: {prediction}")  # Check output probability
        
        return "Healthy" if prediction < 0.5 else "Unhealthy"
    return None

# -------------------- Main App --------------------

def main():
    st.sidebar.title("🌾 AgroBloom AI")
    menu = [
        "Home", 
        "Weather Forecasting", 
        "Irrigation Management", 
        "Soil Health Analysis", 
        "Crop Recommendation", 
        "Crop Health Monitoring"
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
                {"icon": "🌱", "title": "Crop Health Guardian", "desc": "Visual analysis of plant health using computer vision"},
                {"icon": "📊", "title": "Soil Health Dashboard", "desc": "Comprehensive nutrient analysis and recommendations"}
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
            description="AI-powered water management for optimal crop growth",
            color_name="blue-70"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Field Parameters")
            city = st.text_input("📍 Location", "London")
            soil_moisture = st.slider("Current Soil Moisture (%)", 0, 100, 30)
            crop_type = st.selectbox("Crop Type", ["Wheat", "Corn", "Rice", "Soybean"])
            st.caption("💡 Optimal moisture levels vary by crop type")
            
            if st.button("Calculate Irrigation", use_container_width=True):
                with st.spinner("Analyzing field conditions..."):
                    weather_data = get_weather_data(city)
                    if weather_data and weather_data.get("main"):
                        with col2:
                            st.subheader("Irrigation Plan")
                            irrigation_management(weather_data, soil_moisture)
                            
                            # Visual moisture indicator
                            st.markdown(
                                f"""
                                <div class="moisture-gauge">
                                    <div class="moisture-fill" style="width: {soil_moisture}%">
                                        {soil_moisture}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True
                            )
                            
                            # Water saving stats
                            st.markdown(
                                """
                                <div class="savings-card">
                                    <h3>💧 Water Conservation</h3>
                                    <p>Potential savings this month: <strong>12,500L</strong></p>
                                    <p>Estimated yield improvement: <strong>15-20%</strong></p>
                                </div>
                                """, unsafe_allow_html=True
                            )
                    else:
                        st.error("Failed to retrieve weather data for irrigation management.")

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
                                    <li>Apply organic compost as per soil test recommendations.</li>
                                    <li>Ideal Soil pH levels between 6.0 - 7.5.</li>
                                    <li>Ideal Nitrogen levels between 20 - 50</li>
                                    <li>Ideal Phosphorus levels between 15 - 40</li>
                                    <li>Ideal potassium levels between 15 - 40</li>
                                    <li>Retest soil after 45 days.</li>
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

    # -------------------- Crop Health Monitoring --------------------
    elif choice == "Crop Health Monitoring":
        colored_header(
            label="🌿 Crop Health Guardian",
            description="AI-powered plant disease detection and health monitoring",
            color_name="red-70"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Image Analysis")
            uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                os.makedirs("uploaded_images", exist_ok=True)
                file_path = os.path.join("uploaded_images", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.image(file_path, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Crop Health", use_container_width=True):
                    with st.spinner("Scanning for plant diseases..."):
                        # Load the trained crop health model if it exists.
                        if os.path.exists("crop_health_model.h5"):
                            model = load_model("crop_health_model.h5")
                        else:
                            st.info("No crop health model found. Training a placeholder model...")
                            model = train_placeholder_model()
                        health_status = predict_crop_health(model, file_path)
                        with col2:
                            if health_status == "Healthy":
                                st.markdown(
                                    """
                                    <div class="health-result healthy">
                                        <h1>🌱 Healthy Crop!</h1>
                                        <p>No significant diseases detected.</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            elif health_status == "Unhealthy":
                                st.markdown(
                                    """
                                    <div class="health-result unhealthy">
                                        <h1>⚠️ Disease Detected!</h1>
                                        <p>Recommended treatment: Fungicide application and further expert diagnosis.</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            else:
                                st.error("Unable to determine crop health.")

if __name__ == "__main__":
    main()
