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

# -------------------- Multilingual Setup --------------------
# Define a dictionary with translations for English, Hindi, and Marathi.
translations = {
    "en": {
        "sidebar_title": "🌾 AgroBloom AI",
        "home_title": "🌱 AgroBloom AI",
        "welcome_message": "Welcome to AgroBloom AI - Your intelligent farming companion! Harness the power of artificial intelligence to optimize your agricultural operations, increase crop yield, and make data-driven decisions for sustainable farming.",
        "weather_report": "Weather Report for",
        "farming_advisory": "Farming Advisory",
        "soil_moisture_map": "Soil Moisture Map",
        "current_conditions": "Current Conditions",
        "select_language": "Select Language",
        "home": "Home",
        "weather_forecasting": "Weather Forecasting",
        "irrigation_management": "Irrigation Management",
        "soil_health_analysis": "Soil Health Analysis",
        "crop_recommendation": "Crop Recommendation",
        "disease_detection": "Disease Detection",
        "navigation": "Navigation",
        "smart_weather_insights": "Smart Weather Insights",
        "smart_weather_desc": "Get real-time weather data and farming advice based on your location.",
        "ai_irrigation_system": "AI-Powered Irrigation System",
        "ai_irrigation_desc": "Optimize water usage with predictive analytics and real-time soil moisture data.",
        "soil_health_dashboard": "Soil Health Dashboard",
        "soil_health_desc": "Analyze soil parameters and get actionable insights for improvement.",
        "disease_detection_title": "Plant Disease Detection",
        "disease_detection_desc": "Upload plant images to detect diseases and receive treatment recommendations.",
        "join_revolution": "Join the Agricultural Revolution",
        "start_journey": "Start your journey towards smarter farming today!",
        "enter_location": "Enter Location",
        "get_weather_analysis": "Get Weather Analysis",
        "fetching_weather_data": "Fetching weather data...",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "precipitation": "Precipitation",
        "wind_speed": "Wind Speed",
        "frost_alert": "Frost alert: Protect sensitive crops from low temperatures.",
        "cool_weather": "Cool weather: Suitable for most crops with proper care.",
        "optimal_conditions": "Optimal conditions: Ideal weather for crop growth.",
        "heat_stress": "Heat stress: Ensure proper irrigation to mitigate high temperatures.",
        "failed_to_retrieve_weather_data": "Failed to retrieve weather data. Please check the location and try again.",
        "smart_irrigation_system": "Smart Irrigation System",
        "irrigation_desc": "Use AI to manage irrigation efficiently.",
        "field_parameters": "Field Parameters",
        "select_region_category": "Select Region Category",
        "select_location": "Select Location",
        "use_custom_location": "Use Custom Location",
        "enter_custom_location": "Enter Custom Location",
        "crop_type": "Crop Type",
        "optimal_moisture_tip": "Optimal moisture varies by crop type.",
        "calculate_irrigation": "Calculate Irrigation Needs",
        "analyzing_field_conditions": "Analyzing field conditions...",
        "irrigation_plan": "Irrigation Plan",
        "irrigation_analysis": "Irrigation Analysis",
        "soil_moisture": "Soil Moisture",
        "soil_moisture_level": "Soil Moisture Level",
        "crop_specific_analysis": "Crop-Specific Analysis",
        "optimal_soil_moisture_range": "Optimal Soil Moisture Range",
        "below_optimal": "Soil moisture is below optimal for {crop_type}. Consider irrigating.",
        "above_optimal": "Soil moisture is above optimal for {crop_type}. Reduce irrigation.",
        "within_optimal": "Soil moisture is within optimal range for {crop_type}.",
        "water_conservation_impact": "Water Conservation Impact",
        "monthly_savings": "Monthly Savings",
        "water_saved": "Water Saved",
        "cost_reduction": "Cost Reduction",
        "environmental_impact": "Environmental Impact",
        "carbon_footprint_reduction": "Carbon Footprint Reduction",
        "sustainability_score": "Sustainability Score",
        "how_to_use": "How to Use",
        "input_soil_test_results": "Input your soil test results below.",
        "click_analyze_soil_health": "Click 'Analyze Soil Health' to process the data.",
        "receive_recommendations": "Receive tailored recommendations for soil improvement.",
        "soil_parameters": "Soil Parameters",
        "soil_ph": "Soil pH",
        "nitrogen": "Nitrogen (kg/ha)",
        "phosphorus": "Phosphorus (kg/ha)",
        "potassium": "Potassium (kg/ha)",
        "organic_matter": "Organic Matter (%)",
        "analyze_soil_health": "Analyze Soil Health",
        "analyzing_soil_composition": "Analyzing soil composition...",
        "analysis_results": "Analysis Results",
        "recommended_actions": "Recommended Actions",
        "apply_organic_compost": "Apply organic compost to improve organic matter.",
        "ideal_soil_ph": "Adjust pH to 6.0-7.5 if needed.",
        "ideal_nitrogen": "Maintain nitrogen levels between 20-50 kg/ha.",
        "ideal_phosphorus": "Maintain phosphorus levels between 15-40 kg/ha.",
        "ideal_potassium": "Maintain potassium levels between 15-40 kg/ha.",
        "retest_soil": "Retest soil after 3 months.",
        "smart_crop_advisor": "Smart Crop Advisor",
        "crop_advisor_desc": "Get crop recommendations based on soil conditions.",
        "field_conditions": "Field Conditions",
        "get_crop_recommendations": "Get Crop Recommendations",
        "analyzing_optimal_crops": "Analyzing optimal crops...",
        "recommended_crop": "Recommended Crop",
        "expected_yield": "Expected Yield",
        "best_season": "Best Season",
        "upload_plant_image_desc": "Upload an image of your plant for analysis.",
        "click_analyze_disease": "Click 'Analyze Disease' to start the process.",
        "receive_disease_recommendations": "Receive disease detection results and recommendations.",
        "upload_plant_image": "Upload Plant Image",
        "uploaded_image": "Uploaded Image",
        "analyze_disease": "Analyze Disease",
        "analyzing_disease": "Analyzing disease...",
        "disease_detection_results": "Disease Detection Results",
        "consult_expert": "Consult a local agricultural expert for confirmation.",
        "apply_treatment": "Apply recommended treatment as soon as possible.",
        "monitor_health": "Monitor plant health over the next few weeks.",
        "invalid_image_error": "Invalid image file. Please upload a valid JPG or PNG image.",
        "error_occurred": "An error occurred"
    },
    "hi": {
        "sidebar_title": "🌾 एग्रोब्लूम एआई",
        "home_title": "🌱 एग्रोब्लूम एआई",
        "welcome_message": "एग्रोब्लूम एआई में आपका स्वागत है - आपका बुद्धिमान कृषि साथी! कृत्रिम बुद्धिमत्ता की शक्ति का उपयोग करें अपनी कृषि गतिविधियों को अनुकूलित करने, फसल की उपज बढ़ाने और सतत कृषि के लिए डेटा-संचालित निर्णय लेने में।",
        "weather_report": "के लिए मौसम रिपोर्ट",
        "farming_advisory": "कृषि सलाह",
        "soil_moisture_map": "मिट्टी की नमी का मानचित्र",
        "current_conditions": "वर्तमान स्थितियां",
        "select_language": "भाषा चुनें",
        "home": "होम",
        "weather_forecasting": "मौसम पूर्वानुमान",
        "irrigation_management": "सिंचाई प्रबंधन",
        "soil_health_analysis": "मिट्टी स्वास्थ्य विश्लेषण",
        "crop_recommendation": "फसल सिफारिश",
        "disease_detection": "रोग पहचान",
        "navigation": "नेविगेशन",
        "smart_weather_insights": "स्मार्ट मौसम अंतर्दृष्टि",
        "smart_weather_desc": "अपने स्थान के आधार पर वास्तविक समय मौसम डेटा और कृषि सलाह प्राप्त करें।",
        "ai_irrigation_system": "एआई-संचालित सिंचाई प्रणाली",
        "ai_irrigation_desc": "भविष्यवाणी विश्लेषण और वास्तविक समय मिट्टी नमी डेटा के साथ जल उपयोग को अनुकूलित करें।",
        "soil_health_dashboard": "मिट्टी स्वास्थ्य डैशबोर्ड",
        "soil_health_desc": "मिट्टी के मापदंडों का विश्लेषण करें और सुधार के लिए उपयोगी जानकारी प्राप्त करें।",
        "disease_detection_title": "पौधे रोग पहचान",
        "disease_detection_desc": "पौधों की छवियां अपलोड करें रोगों का पता लगाने और उपचार सिफारिशें प्राप्त करने के लिए।",
        "join_revolution": "कृषि क्रांति में शामिल हों",
        "start_journey": "आज ही स्मार्ट खेती की ओर अपनी यात्रा शुरू करें!",
        "enter_location": "स्थान दर्ज करें",
        "get_weather_analysis": "मौसम विश्लेषण प्राप्त करें",
        "fetching_weather_data": "मौसम डेटा प्राप्त कर रहा है...",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "precipitation": "वर्षा",
        "wind_speed": "हवा की गति",
        "frost_alert": "पाला चेतावनी: संवेदनशील फसलों को कम तापमान से बचाएं।",
        "cool_weather": "ठंडा मौसम: उचित देखभाल के साथ अधिकांश फसलों के लिए उपयुक्त।",
        "optimal_conditions": "इष्टतम स्थितियां: फसल वृद्धि के लिए आदर्श मौसम।",
        "heat_stress": "गर्मी तनाव: उच्च तापमान को कम करने के लिए उचित सिंचाई सुनिश्चित करें।",
        "failed_to_retrieve_weather_data": "मौसम डेटा प्राप्त करने में विफल। कृपया स्थान जांचें और पुनः प्रयास करें।",
        "smart_irrigation_system": "स्मार्ट सिंचाई प्रणाली",
        "irrigation_desc": "सिंचाई को कुशलतापूर्वक प्रबंधित करने के लिए एआई का उपयोग करें।",
        "field_parameters": "खेत के मापदंड",
        "select_region_category": "क्षेत्र श्रेणी चुनें",
        "select_location": "स्थान चुनें",
        "use_custom_location": "कस्टम स्थान का उपयोग करें",
        "enter_custom_location": "कस्टम स्थान दर्ज करें",
        "crop_type": "फसल प्रकार",
        "optimal_moisture_tip": "इष्टतम नमी फसल प्रकार के अनुसार भिन्न होती है।",
        "calculate_irrigation": "सिंचाई की आवश्यकता की गणना करें",
        "analyzing_field_conditions": "खेत की स्थितियों का विश्लेषण कर रहा है...",
        "irrigation_plan": "सिंचाई योजना",
        "irrigation_analysis": "सिंचाई विश्लेषण",
        "soil_moisture": "मिट्टी की नमी",
        "soil_moisture_level": "मिट्टी की नमी स्तर",
        "crop_specific_analysis": "फसल-विशिष्ट विश्लेषण",
        "optimal_soil_moisture_range": "इष्टतम मिट्टी नमी रेंज",
        "below_optimal": "{crop_type} के लिए मिट्टी की नमी इष्टतम से कम है। सिंचाई पर विचार करें।",
        "above_optimal": "{crop_type} के लिए मिट्टी की नमी इष्टतम से अधिक है। सिंचाई कम करें।",
        "within_optimal": "{crop_type} के लिए मिट्टी की नमी इष्टतम रेंज में है।",
        "water_conservation_impact": "जल संरक्षण प्रभाव",
        "monthly_savings": "मासिक बचत",
        "water_saved": "पानी बचाया गया",
        "cost_reduction": "लागत में कमी",
        "environmental_impact": "पर्यावरणीय प्रभाव",
        "carbon_footprint_reduction": "कार्बन फुटप्रिंट में कमी",
        "sustainability_score": "स्थिरता स्कोर",
        "how_to_use": "उपयोग कैसे करें",
        "input_soil_test_results": "नीचे अपने मिट्टी परीक्षण परिणाम दर्ज करें।",
        "click_analyze_soil_health": "डेटा को संसाधित करने के लिए 'मिट्टी स्वास्थ्य विश्लेषण' पर क्लिक करें।",
        "receive_recommendations": "मिट्टी सुधार के लिए अनुकूलित सिफारिशें प्राप्त करें।",
        "soil_parameters": "मिट्टी के मापदंड",
        "soil_ph": "मिट्टी का पीएच",
        "nitrogen": "नाइट्रोजन (किग्रा/हेक्टेयर)",
        "phosphorus": "फॉस्फोरस (किग्रा/हेक्टेयर)",
        "potassium": "पोटैशियम (किग्रा/हेक्टेयर)",
        "organic_matter": "जैविक पदार्थ (%)",
        "analyze_soil_health": "मिट्टी स्वास्थ्य विश्लेषण",
        "analyzing_soil_composition": "मिट्टी संरचना का विश्लेषण कर रहा है...",
        "analysis_results": "विश्लेषण परिणाम",
        "recommended_actions": "अनुशंसित कार्रवाइयाँ",
        "apply_organic_compost": "जैविक खाद लागू करें ताकि जैविक पदार्थ में सुधार हो।",
        "ideal_soil_ph": "यदि आवश्यक हो तो पीएच को 6.0-7.5 तक समायोजित करें।",
        "ideal_nitrogen": "नाइट्रोजन स्तर को 20-50 किग्रा/हेक्टेयर के बीच बनाए रखें।",
        "ideal_phosphorus": "फॉस्फोरस स्तर को 15-40 किग्रा/हेक्टेयर के बीच बनाए रखें।",
        "ideal_potassium": "पोटैशियम स्तर को 15-40 किग्रा/हेक्टेयर के बीच बनाए रखें।",
        "retest_soil": "3 महीने बाद मिट्टी का पुनः परीक्षण करें।",
        "smart_crop_advisor": "स्मार्ट फसल सलाहकार",
        "crop_advisor_desc": "मिट्टी की स्थिति के आधार पर फसल सिफारिशें प्राप्त करें।",
        "field_conditions": "खेत की स्थितियाँ",
        "get_crop_recommendations": "फसल सिफारिशें प्राप्त करें",
        "analyzing_optimal_crops": "इष्टतम फसलों का विश्लेषण कर रहा है...",
        "recommended_crop": "अनुशंसित फसल",
        "expected_yield": "अपेक्षित उपज",
        "best_season": "सर्वश्रेष्ठ मौसम",
        "upload_plant_image_desc": "विश्लेषण के लिए अपने पौधे की छवि अपलोड करें।",
        "click_analyze_disease": "प्रक्रिया शुरू करने के लिए 'रोग विश्लेषण' पर क्लिक करें।",
        "receive_disease_recommendations": "रोग पहचान परिणाम और सिफारिशें प्राप्त करें।",
        "upload_plant_image": "पौधे की छवि अपलोड करें",
        "uploaded_image": "अपलोड की गई छवि",
        "analyze_disease": "रोग विश्लेषण",
        "analyzing_disease": "रोग का विश्लेषण कर रहा है...",
        "disease_detection_results": "रोग पहचान परिणाम",
        "consult_expert": "पुष्टि के लिए स्थानीय कृषि विशेषज्ञ से परामर्श करें।",
        "apply_treatment": "जल्द से जल्द अनुशंसित उपचार लागू करें।",
        "monitor_health": "अगले कुछ हफ्तों तक पौधे के स्वास्थ्य की निगरानी करें।",
        "invalid_image_error": "अमान्य छवि फ़ाइल। कृपया एक मान्य JPG या PNG छवि अपलोड करें।",
        "error_occurred": "एक त्रुटि हुई"
    },
    "mr": {
        "sidebar_title": "🌾 एग्रोब्लूम एआई",
        "home_title": "🌱 एग्रोब्लूम एआई",
        "welcome_message": "एग्रोब्लूम एआई मध्ये आपले स्वागत आहे - आपला बुद्धिमान शेती साथी! कृत्रिम बुद्धिमत्ता चा वापर करून आपल्या कृषी कार्यांची कार्यक्षमता वाढवा, पीक उत्पादन वाढवा आणि शाश्वत शेतीसाठी डेटा-आधारित निर्णय घ्या.",
        "weather_report": "साठी हवामान अहवाल",
        "farming_advisory": "शेती सल्ला",
        "soil_moisture_map": "मातीची आर्द्रता नकाशा",
        "current_conditions": "सध्याच्या परिस्थिती",
        "select_language": "भाषा निवडा",
        "home": "मुखपृष्ठ",
        "weather_forecasting": "हवामान अंदाज",
        "irrigation_management": "सिंचन व्यवस्थापन",
        "soil_health_analysis": "माती आरोग्य विश्लेषण",
        "crop_recommendation": "पीक शिफारस",
        "disease_detection": "रोग शोध",
        "navigation": "नेव्हिगेशन",
        "smart_weather_insights": "स्मार्ट हवामान अंतर्दृष्टी",
        "smart_weather_desc": "आपल्या स्थानावर आधारित वास्तविक वेळ हवामान डेटा आणि शेती सल्ला मिळवा.",
        "ai_irrigation_system": "एआय-चालित सिंचन प्रणाली",
        "ai_irrigation_desc": "भविष्यसूचक विश्लेषण आणि वास्तविक वेळ माती आर्द्रता डेटा सह पाण्याचा वापर अनुकूल करा.",
        "soil_health_dashboard": "माती आरोग्य डॅशबोर्ड",
        "soil_health_desc": "मातीच्या मापदंडांचे विश्लेषण करा आणि सुधारणेसाठी उपयुक्त अंतर्दृष्टी मिळवा.",
        "disease_detection_title": "वनस्पती रोग शोध",
        "disease_detection_desc": "रोग शोधण्यासाठी आणि उपचार शिफारशी मिळवण्यासाठी वनस्पतींची छायाचित्रे अपलोड करा.",
        "join_revolution": "कृषी क्रांतीत सामील व्हा",
        "start_journey": "आजच स्मार्ट शेतीकडे आपली यात्रा सुरू करा!",
        "enter_location": "स्थान प्रविष्ट करा",
        "get_weather_analysis": "हवामान विश्लेषण मिळवा",
        "fetching_weather_data": "हवामान डेटा मिळवत आहे...",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "precipitation": "पर्जन्यमान",
        "wind_speed": "वाऱ्याचा वेग",
        "frost_alert": "हिम चेतावनी: संवेदनशील पिकांना कमी तापमानापासून संरक्षण द्या.",
        "cool_weather": "थंड हवामान: योग्य काळजीसह बहुतेक पिकांसाठी योग्य.",
        "optimal_conditions": "इष्टतम परिस्थिती: पीक वाढीसाठी आदर्श हवामान.",
        "heat_stress": "उष्णता तणाव: उच्च तापमान कमी करण्यासाठी योग्य सिंचन सुनिश्चित करा.",
        "failed_to_retrieve_weather_data": "हवामान डेटा मिळवण्यात अयशस्वी. कृपया स्थान तपासा आणि पुन्हा प्रयत्न करा.",
        "smart_irrigation_system": "स्मार्ट सिंचन प्रणाली",
        "irrigation_desc": "सिंचन कार्यक्षमतेने व्यवस्थापित करण्यासाठी एआयचा वापर करा.",
        "field_parameters": "शेत मापदंड",
        "select_region_category": "प्रदेश श्रेणी निवडा",
        "select_location": "स्थान निवडा",
        "use_custom_location": "सानुकूल स्थान वापरा",
        "enter_custom_location": "सानुकूल स्थान प्रविष्ट करा",
        "crop_type": "पीक प्रकार",
        "optimal_moisture_tip": "इष्टतम आर्द्रता पीक प्रकारानुसार बदलते.",
        "calculate_irrigation": "सिंचनाची गरज मोजा",
        "analyzing_field_conditions": "शेताच्या परिस्थितीचे विश्लेषण करत आहे...",
        "irrigation_plan": "सिंचन योजना",
        "irrigation_analysis": "सिंचन विश्लेषण",
        "soil_moisture": "मातीची आर्द्रता",
        "soil_moisture_level": "मातीची आर्द्रता पातळी",
        "crop_specific_analysis": "पीक-विशिष्ट विश्लेषण",
        "optimal_soil_moisture_range": "इष्टतम माती आर्द्रता श्रेणी",
        "below_optimal": "{crop_type} साठी मातीची आर्द्रता इष्टतमपेक्षा कमी आहे. सिंचनाचा विचार करा.",
        "above_optimal": "{crop_type} साठी मातीची आर्द्रता इष्टतमपेक्षा जास्त आहे. सिंचन कमी करा.",
        "within_optimal": "{crop_type} साठी मातीची आर्द्रता इष्टतम श्रेणीत आहे.",
        "water_conservation_impact": "जल संरक्षण प्रभाव",
        "monthly_savings": "मासिक बचत",
        "water_saved": "पाणी वाचले",
        "cost_reduction": "खर्चात कपात",
        "environmental_impact": "पर्यावरणीय प्रभाव",
        "carbon_footprint_reduction": "कार्बन फूटप्रिंट कमी करणे",
        "sustainability_score": "शाश्वतता गुण",
        "how_to_use": "कसे वापरावे",
        "input_soil_test_results": "खाली आपले माती चाचणी परिणाम प्रविष्ट करा.",
        "click_analyze_soil_health": "डेटा प्रक्रिया करण्यासाठी 'माती आरोग्य विश्लेषण' वर क्लिक करा.",
        "receive_recommendations": "माती सुधारणेसाठी सानुकूलित शिफारशी मिळवा.",
        "soil_parameters": "माती मापदंड",
        "soil_ph": "माती पीएच",
        "nitrogen": "नायट्रोजन (किग्रा/हेक्टर)",
        "phosphorus": "फॉस्फरस (किग्रा/हेक्टर)",
        "potassium": "पोटॅशियम (किग्रा/हेक्टर)",
        "organic_matter": "सेंद्रिय पदार्थ (%)",
        "analyze_soil_health": "माती आरोग्य विश्लेषण",
        "analyzing_soil_composition": "माती संरचनेचे विश्लेषण करत आहे...",
        "analysis_results": "विश्लेषण परिणाम",
        "recommended_actions": "शिफारस केलेल्या कृती",
        "apply_organic_compost": "सेंद्रिय खाद लावा जेणेकरून सेंद्रिय पदार्थात सुधारणा होईल.",
        "ideal_soil_ph": "आवश्यक असल्यास पीएच 6.0-7.5 पर्यंत समायोजित करा.",
        "ideal_nitrogen": "नायट्रोजन पातळी 20-50 किग्रा/हेक्टर दरम्यान ठेवा.",
        "ideal_phosphorus": "फॉस्फरस पातळी 15-40 किग्रा/हेक्टर दरम्यान ठेवा.",
        "ideal_potassium": "पोटॅशियम पातळी 15-40 किग्रा/हेक्टर दरम्यान ठेवा.",
        "retest_soil": "3 महिन्यांनंतर मातीची पुन्हा चाचणी करा.",
        "smart_crop_advisor": "स्मार्ट पीक सल्लागार",
        "crop_advisor_desc": "मातीच्या परिस्थितीवर आधारित पीक शिफारशी मिळवा.",
        "field_conditions": "शेत परिस्थिती",
        "get_crop_recommendations": "पीक शिफारशी मिळवा",
        "analyzing_optimal_crops": "इष्टतम पिकांचे विश्लेषण करत आहे...",
        "recommended_crop": "शिफारस केलेले पीक",
        "expected_yield": "अपेक्षित उत्पादन",
        "best_season": "सर्वोत्तम हंगाम",
        "upload_plant_image_desc": "विश्लेषणासाठी आपल्या वनस्पतीचे चित्र अपलोड करा.",
        "click_analyze_disease": "प्रक्रिया सुरू करण्यासाठी 'रोग विश्लेषण' वर क्लिक करा.",
        "receive_disease_recommendations": "रोग शोध परिणाम आणि शिफारशी मिळवा.",
        "upload_plant_image": "वनस्पती चित्र अपलोड करा",
        "uploaded_image": "अपलोड केलेले चित्र",
        "analyze_disease": "रोग विश्लेषण",
        "analyzing_disease": "रोगाचे विश्लेषण करत आहे...",
        "disease_detection_results": "रोग शोध परिणाम",
        "consult_expert": "पुष्टीकरणासाठी स्थानिक कृषी तज्ञाचा सल्ला घ्या.",
        "apply_treatment": "शक्य तितक्या लवकर शिफारस केलेले उपचार लागू करा.",
        "monitor_health": "पुढील काही आठवड्यांपर्यंत वनस्पतीच्या आरोग्यावर लक्ष ठेवा.",
        "invalid_image_error": "अवैध चित्र फाइल. कृपया वैध JPG किंवा PNG चित्र अपलोड करा.",
        "error_occurred": "एक त्रुटी आली"
    }
}
# Global variable to store the current language code
current_language = "en"

# Helper function to return translated text for a given key
def tr(key):
    return translations.get(current_language, translations["en"]).get(key, key)


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
    global current_language

    # Language selection
    language = st.sidebar.selectbox(tr("select_language"), ["English", "हिंदी", "मराठी"])
    if language == "English":
        current_language = "en"
    elif language == "हिंदी":
        current_language = "hi"
    elif language == "मराठी":
        current_language = "mr"

    # Sidebar
    st.sidebar.title(tr("sidebar_title"))
    menu = [
        tr("home"),
        tr("weather_forecasting"),
        tr("irrigation_management"),
        tr("soil_health_analysis"),
        tr("crop_recommendation"),
        tr("disease_detection")
    ]
    choice = st.sidebar.radio(tr("navigation"), menu)

    # Home Page
    if choice == tr("home"):
        st.markdown(f"<h1 style='text-align: center; color: #2E8B57;'>{tr('home_title')}</h1>", unsafe_allow_html=True)
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""
                <div style='text-align: justify; font-size: 20px;'>
                    {tr('welcome_message')}
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            features = [
                {"icon": "🌤️", "title": tr("smart_weather_insights"), "desc": tr("smart_weather_desc")},
                {"icon": "💧", "title": tr("ai_irrigation_system"), "desc": tr("ai_irrigation_desc")},
                {"icon": "📊", "title": tr("soil_health_dashboard"), "desc": tr("soil_health_desc")},
                {"icon": "🌿", "title": tr("disease_detection_title"), "desc": tr("disease_detection_desc")}
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
        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px;'>
                <h3>🌍 {tr('join_revolution')}</h3>
                <p>{tr('start_journey')}</p>
            </div>
            """, unsafe_allow_html=True
        )

    # Weather Forecasting
    elif choice == tr("weather_forecasting"):
        colored_header(
            label=tr("weather_report"),
            description=tr("smart_weather_desc"),
            color_name="green-70"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            city = st.text_input(f"📍 {tr('enter_location')}", "London")
            if st.button(tr("get_weather_analysis"), use_container_width=True):
                with st.spinner(tr("fetching_weather_data")):
                    weather_data = get_weather_data(city)
                    if weather_data and weather_data.get("main"):
                        with col2:
                            st.subheader(f"{tr('weather_report')} {city}")
                            cols = st.columns(4)
                            cols[0].metric(tr("temperature"), f"{weather_data['main']['temp']}°C",
                                           help="Optimal range for most crops: 15-30°C")
                            cols[1].metric(tr("humidity"), f"{weather_data['main']['humidity']}%",
                                           "Ideal range: 40-80%")
                            cols[2].metric(tr("precipitation"),
                                           f"{weather_data.get('rain', {}).get('1h', 0)}mm",
                                           "Next 3 hours")
                            cols[3].metric(tr("wind_speed"),
                                           f"{weather_data['wind']['speed']} m/s",
                                           "Wind direction")
                            style_metric_cards()

                            temp = weather_data['main']['temp']
                            if temp < 10:
                                advisory = tr("frost_alert")
                            elif 10 <= temp < 20:
                                advisory = tr("cool_weather")
                            elif 20 <= temp < 30:
                                advisory = tr("optimal_conditions")
                            else:
                                advisory = tr("heat_stress")

                            st.markdown(
                                f"""
                                <div class="advisory-box">
                                    <h3>🌱 {tr('farming_advisory')}</h3>
                                    <p>{advisory}</p>
                                </div>
                                """, unsafe_allow_html=True
                            )
                    else:
                        st.error(tr("failed_to_retrieve_weather_data"))

    # Irrigation Management
    elif choice == tr("irrigation_management"):
        colored_header(
            label=tr("smart_irrigation_system"),
            description=tr("irrigation_desc"),
            color_name="blue-70"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(tr("field_parameters"))

            region_category = st.selectbox(tr("select_region_category"), ["India", "United States", "Europe"])
            city = st.selectbox(f"📍 {tr('select_location')}", suggested_regions[region_category])

            use_custom_location = st.checkbox(tr("use_custom_location"))
            if use_custom_location:
                city = st.text_input(tr("enter_custom_location"))

            crop_type = st.selectbox(tr("crop_type"), ["Wheat", "Corn", "Rice", "Soybean"])
            st.caption(tr("optimal_moisture_tip"))

            if st.button(tr("calculate_irrigation"), use_container_width=True):
                with st.spinner(tr("analyzing_field_conditions")):
                    lat, lon = get_coordinates(city)
                    if lat and lon:
                        soil_moisture = get_soil_moisture(lat, lon)
                        if soil_moisture is not None:
                            weather_data = get_weather_data(city)
                            if weather_data and weather_data.get("main"):
                                with col2:
                                    st.subheader(tr("irrigation_plan"))

                                    st.write(f"### 🗺️ {tr('soil_moisture_map')}")
                                    display_soil_moisture_map(lat, lon, soil_moisture)

                                    st.write(f"### 📊 {tr('current_conditions')}")
                                    cols = st.columns(3)
                                    cols[0].metric(
                                        tr("soil_moisture"),
                                        f"{soil_moisture}%",
                                        delta="Real-time data",
                                        help="Real-time soil moisture from satellite data"
                                    )
                                    cols[1].metric(
                                        tr("temperature"),
                                        f"{weather_data['main']['temp']}°C",
                                        help=tr("current_temperature")
                                    )
                                    cols[2].metric(
                                        tr("humidity"),
                                        f"{weather_data['main']['humidity']}%",
                                        help=tr("current_humidity")
                                    )

                                    st.write(f"### 💧 {tr('irrigation_analysis')}")
                                    irrigation_management(weather_data, soil_moisture)

                                    st.markdown(
                                        f"""
                                        <div style="margin: 20px 0;">
                                            <h4>{tr('soil_moisture_level')}</h4>
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

                                    optimal_moisture = {
                                        "Wheat": (30, 50),
                                        "Corn": (35, 55),
                                        "Rice": (60, 85),
                                        "Soybean": (40, 60)
                                    }
                                    crop_range = optimal_moisture[crop_type]
                                    st.write(f"### 🌾 {tr('crop_specific_analysis')} {crop_type}")
                                    st.write(f"{tr('optimal_soil_moisture_range')}: {crop_range[0]}% - {crop_range[1]}%")

                                    if soil_moisture < crop_range[0]:
                                        st.warning(tr("below_optimal").format(crop_type=crop_type))
                                    elif soil_moisture > crop_range[1]:
                                        st.warning(tr("above_optimal").format(crop_type=crop_type))
                                    else:
                                        st.success(tr("within_optimal").format(crop_type=crop_type))

                                    st.markdown(
                                        f"""
                                        <div style="
                                            background: #f8f9fa;
                                            border-radius: 10px;
                                            padding: 20px;
                                            margin-top: 20px;
                                        ">
                                            <h3>{tr('water_conservation_impact')}</h3>
                                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                                <div>
                                                    <h4>{tr('monthly_savings')}</h4>
                                                    <p>{tr('water_saved')}: <strong>12,500L</strong></p>
                                                    <p>{tr('cost_reduction')}: <strong>15%</strong></p>
                                                </div>
                                                <div>
                                                    <h4>{tr('environmental_impact')}</h4>
                                                    <p>{tr('carbon_footprint_reduction')}: <strong>25%</strong></p>
                                                    <p>{tr('sustainability_score')}: <strong>8.5/10</strong></p>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )

    # Soil Health Analysis
    elif choice == tr("soil_health_analysis"):
        colored_header(
            label=tr("soil_health_dashboard"),
            description=tr("soil_health_desc"),
            color_name="orange-70"
        )

        with st.expander(f"🔍 {tr('how_to_use')}"):
            st.markdown(
                f"""
                1. {tr('input_soil_test_results')}<br>
                2. {tr('click_analyze_soil_health')}<br>
                3. {tr('receive_recommendations')}
                """, unsafe_allow_html=True
            )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(tr("soil_parameters"))
            pH = st.slider(tr("soil_ph"), 0.0, 14.0, 7.0, 0.1)
            nitrogen = st.number_input(tr("nitrogen"), min_value=0.0, value=20.0)
            phosphorus = st.number_input(tr("phosphorus"), min_value=0.0, value=15.0)
            potassium = st.number_input(tr("potassium"), min_value=0.0, value=15.0)
            organic_matter = st.slider(tr("organic_matter"), 0.0, 100.0, 5.0)

            if st.button(tr("analyze_soil_health"), use_container_width=True):
                with st.spinner(tr("analyzing_soil_composition")):
                    health_status = analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter)
                    with col2:
                        st.subheader(tr("analysis_results"))
                        st.markdown(
                            f"""
                            <div class="gauge-container">
                                <p><strong>pH:</strong> {health_status['pH']}</p>
                                <p><strong>{tr('nitrogen')}:</strong> {health_status['Nitrogen']}</p>
                                <p><strong>{tr('phosphorus')}:</strong> {health_status['Phosphorus']}</p>
                                <p><strong>{tr('potassium')}:</strong> {health_status['Potassium']}</p>
                                <p><strong>{tr('organic_matter')}:</strong> {health_status['Organic Matter']}</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.markdown(
                            f"""
                            <div class="recommendation-box">
                                <h3>{tr('recommended_actions')}</h3>
                                <ul>
                                    <li>{tr('apply_organic_compost')}</li>
                                    <li>{tr('ideal_soil_ph')}</li>
                                    <li>{tr('ideal_nitrogen')}</li>
                                    <li>{tr('ideal_phosphorus')}</li>
                                    <li>{tr('ideal_potassium')}</li>
                                    <li>{tr('retest_soil')}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True
                        )

    # Crop Recommendation
    elif choice == tr("crop_recommendation"):
        colored_header(
            label=tr("smart_crop_advisor"),
            description=tr("crop_advisor_desc"),
            color_name="violet-70"
        )

        soil_data, crop_production_data = load_data()
        if soil_data is not None and crop_production_data is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader(tr("field_conditions"))
                pH = st.slider(tr("soil_ph"), 0.0, 14.0, 7.0, 0.1)
                nitrogen = st.number_input(tr("nitrogen"), min_value=0.0, value=20.0)
                phosphorus = st.number_input(tr("phosphorus"), min_value=0.0, value=15.0)
                potassium = st.number_input(tr("potassium"), min_value=0.0, value=15.0)
                organic_matter = st.slider(tr("organic_matter"), 0.0, 100.0, 5.0)

                if st.button(tr("get_crop_recommendations"), use_container_width=True):
                    with st.spinner(tr("analyzing_optimal_crops")):
                        model = load_crop_recommendation_model()
                        if model:
                            soil_data_row = [pH, nitrogen, phosphorus, potassium, organic_matter]
                            recommended_crop = recommend_crops_with_model(model, soil_data_row)
                            with col2:
                                st.markdown(
                                    f"""
                                    <div class="crop-card">
                                        <h2>{tr('recommended_crop')}</h2>
                                        <h1 class="crop-name">🌽 {recommended_crop}</h1>
                                        <div class="crop-stats">
                                            <div>
                                                <h3>{tr('expected_yield')}</h3>
                                                <p>12-15 tons/ha</p>
                                            </div>
                                            <div>
                                                <h3>{tr('best_season')}</h3>
                                                <p>Kharif/Rabi</p>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True
                                )

    # Disease Detection
    elif choice == tr("disease_detection"):
        colored_header(
            label=tr("disease_detection_title"),
            description=tr("disease_detection_desc"),
            color_name="red-70"
        )

        with st.expander(f"🔍 {tr('how_to_use')}"):
            st.markdown(
                f"""
                1. {tr('upload_plant_image_desc')}<br>
                2. {tr('click_analyze_disease')}<br>
                3. {tr('receive_disease_recommendations')}
                """, unsafe_allow_html=True
            )

        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_file = st.file_uploader(tr("upload_plant_image"), type=["jpg", "png"])
            if uploaded_file:
                try:
                    image = Image.open(BytesIO(uploaded_file.read()))
                    st.image(image, caption=tr("uploaded_image"), use_column_width=True)

                    if st.button(tr("analyze_disease"), use_container_width=True):
                        with st.spinner(tr("analyzing_disease")):
                            disease_info = analyze_plant_disease(image)
                            with col2:
                                st.subheader(tr("disease_detection_results"))
                                st.write(disease_info)
                                st.markdown(
                                    f"""
                                    <div class="recommendation-box">
                                        <h3>{tr('recommended_actions')}</h3>
                                        <ul>
                                            <li>{tr('consult_expert')}</li>
                                            <li>{tr('apply_treatment')}</li>
                                            <li>{tr('monitor_health')}</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                except UnidentifiedImageError:
                    st.error(tr("invalid_image_error"))
                except Exception as e:
                    st.error(f"{tr('error_occurred')}: {str(e)}")

if __name__ == "__main__":
    main()
