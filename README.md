# 🌾 Agriculture Management System (ML-Powered)

## 🔍 Overview
The **Agriculture Management System** is an AI-powered platform that helps farmers make data-driven decisions by providing:
- 🌱 **Crop Recommendation** based on soil health, weather conditions, and historical data.
- 💧 **Irrigation Management** using real-time moisture levels and weather predictions.
- 🌦️ **Weather Forecasting** for optimizing farming schedules.
- 🌾 **Crop Health Analyses** for predicting the crop health based on crop image.

## 🚀 Features
- ✅ **Machine Learning-Based Crop Recommendations**
- ✅ **Weather and Soil Moisture Analysis**
- ✅ **Farmer-Friendly Interactive Dashboard**
- ✅ **Irrigation Management System**
- ✅ **Multilingual Support for Farmers**

## 🛠️ Tech Stack
🌐 **Frontend:**
- Streamlit – Web-based UI for ML model interaction
  
🖥️ **Backend:**
- Python (Streamlit) – Serving ML models and handling UI
  
🤖 **Machine Learning:**
- Scikit-Learn – Random Forest models for classification & regression
- TensorFlow/Keras – Deep learning (CNN) for image processing

📡 **APIs & Integrations:**
- OpenWeather API – Fetches real-time weather data
- Requests & JSON – Handles API calls

🔧 **Tools & Libraries:**
- OpenCV (cv2) – Image processing for crop health analysis
- Pandas & NumPy – Data manipulation
- Joblib – Model saving/loading
- Pillow (PIL) – Image handling
- Streamlit Extras – UI enhancements (colored_header, metric_cards)

## 📂 Project Structure
Agriculture-Management-ML/
│── app.py                    # Main Streamlit application
│── crop_recommendation.py     # Crop Recommendation Model
│── irrigation_model.py        # Irrigation Prediction Model
│── crop_recommendation_model.pkl # Trained Crop Model (Extract the zip)
│── irrigation_model.h5         # Trained Irrigation Model
│── soil_data.csv              # Soil Dataset
│── weather_data.csv           # Weather Dataset
│── project_overview.md        # Documentation for the project
│── uploaded_images            # Images (Extract the zip)
│── .gitignore                 # Ignore unnecessary files
│── README.md                  # Main project description
│── requirements.txt           # Dependencies for ML model
│── LICENSE                    # Open-source license
