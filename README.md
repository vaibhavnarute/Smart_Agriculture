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
**🌐 Frontend:**
- Streamlit – Web-based UI for ML model interaction
  
**🖥️ Backend:**
- Python (Streamlit) – Serving ML models and handling UI
  
**🤖 Machine Learning:**
- Scikit-Learn – Random Forest models for classification & regression
- TensorFlow/Keras – Deep learning (CNN) for image processing

**📡 APIs & Integrations:**
- OpenWeather API – Fetches real-time weather data
- Requests & JSON – Handles API calls

**🔧 Tools & Libraries:**
- OpenCV (cv2) – Image processing for crop health analysis
- Pandas & NumPy – Data manipulation
- Joblib – Model saving/loading
- Pillow (PIL) – Image handling
- Streamlit Extras – UI enhancements (colored_header, metric_cards)

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/a7eb689e-1988-4b38-b2c0-4cddedf974c4)
#
![Screenshot 2025-02-19 120657](https://github.com/user-attachments/assets/b8504e67-2c23-4673-b6d6-94d974960017)
#
![Screenshot 2025-02-19 120718](https://github.com/user-attachments/assets/cc112a3b-8cde-43fb-825d-e7c22d8c5e6d)
#
![Screenshot 2025-02-19 120743](https://github.com/user-attachments/assets/b60106d7-5890-4fb0-9934-acf47080d621)


## ⚙️ Installation Guide
### 1️⃣ Clone the repository  
```sh
git clone https://github.com/yourusername/Agriculture-Management-ML.git
cd Agriculture-Management-ML
```

### 2️⃣ Install Backend Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Install Frontend Dependencies
```sh
npm install
```

### 4️⃣ Run the Project
- Backend
```sh
python app.py
```
- Frontend
```sh
npm start
```

## 🔥 Machine Learning Model Training
To train the machine learning models used in this project, run the following command:
```sh
python scripts/train_model.py
```
**Dataset Details:**
- Soil Data: Measurements including soil pH, nitrogen, phosphorus, potassium levels, and organic matter percentage.
- Weather Data: Historical records with temperature, humidity, and precipitation.
- Crop Production Data: Data on crop yields and types across different regions.

**Model Training Process**
- Data Preprocessing:
  Load and clean the soil, weather, and crop production datasets.
  Merge datasets to create a comprehensive training set.
- Model Selection:
  Crop Recommendation: Utilizes a RandomForestClassifier to suggest optimal crops based on soil parameters.
  Irrigation Management: Uses a RandomForestRegressor to predict soil moisture levels.
  Crop Health Monitoring: Trains a CNN model on image data to classify crop health.
- Training & Evaluation:
  Split data into training and testing sets.
  Train the models and evaluate their performance (e.g., model accuracy for classification tasks).
  Save the trained models (e.g., crop_recommendation_model.pkl, irrigation_model.h5, and crop_health_model.h5).
  
## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
For any queries, contact me at swarupkakade1810@gmail.com or connect on LinkedIn: https://www.linkedin.com/in/swarup1109/

## We hope AgroBloom-AI helps revolutionize your farming practices. Happy farming!
```css
This README provides a comprehensive overview of Agriculture Management System by features and tech stack to detailed installation, usage, and training instructions.
```
