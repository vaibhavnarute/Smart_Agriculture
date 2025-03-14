# AgroBloom AI - Smart Agriculture Management System

An intelligent farming companion that uses AI to help optimize agricultural operations, increase crop yield, and make data-driven decisions for sustainable farming.

## Features

- 🌤️ **Smart Weather Insights**: Real-time weather predictions and adaptive planning
- 💧 **AI Irrigation System**: Optimized water usage with predictive analytics
- 📊 **Soil Health Dashboard**: Comprehensive nutrient analysis and recommendations
- 🌿 **Disease Detection**: AI-powered plant disease detection and treatment recommendations
- 🌾 **Crop Recommendation**: Smart crop suggestions based on soil conditions

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Agriculture-Management-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your API keys:
     ```
     API_KEY=your_openweather_api_key_here
     GOOGLE_API_KEY=your_google_gemini_api_key_here
     ```
   - Get your OpenWeather API key from: https://openweathermap.org/api
   - Get your Google Gemini API key from: https://makersuite.google.com/app/apikey

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Weather Forecasting**:
   - Enter your location
   - Get real-time weather insights and farming recommendations

2. **Irrigation Management**:
   - Input field parameters
   - Receive AI-powered irrigation recommendations

3. **Soil Health Analysis**:
   - Enter soil test results
   - Get comprehensive soil health analysis and recommendations

4. **Disease Detection**:
   - Upload plant images
   - Receive disease diagnosis and treatment recommendations

5. **Crop Recommendation**:
   - Input soil parameters
   - Get AI-powered crop suggestions

## Requirements

- Python 3.8+
- Internet connection for API access
- Valid API keys for OpenWeather and Google Gemini

## Data Files

Make sure you have the following data files in your directory:
- soil_analysis_data.csv
- crop_production_data.csv

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.