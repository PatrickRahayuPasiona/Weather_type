import streamlit as st
import joblib 
import numpy as np
import pandas as pd

model=joblib.load('dtc_model.pkl')
label_to_weather_type = {0: 'Cloudy', 1: 'Rainy', 2: 'Snowy', 3: 'Sunny'}
weather_type_to_image={
    'Cloudy':'cloudy.jpg',
    'Rainy':'rainy.jpg',
    'Snowy':'snowy.jpg',
    'Sunny':'sunny.jpg'
}


def main():
    st.title("Weather Type Prediction App")
    
    #Users of my app will type here
    st.header("Input Features")
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
    precipitation = st.number_input("Precipitation (%)", min_value=0.0, max_value=100.0, value=50.0)
    cloud_cover = st.selectbox("Cloud Cover", ["clear", "cloudy", "overcast", "partly cloudy"])
    atmospheric_pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=870.0, max_value=1100.0, value=1013.0)
    uv_index_value = st.number_input("UV Index", min_value=0.0, max_value=15.0, value=5.0)    
    season = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])
    visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=20.0, value=10.0)
    location = st.selectbox("Location", ["coastal", "inland", "mountain"])
    
    # Calculate heat_index and wind_chill
    heat_index = -42.379 + 2.04901523 * temperature + 10.14333127 * humidity - 0.22475541 * temperature * humidity - 6.83783e-03 * temperature**2 - 5.481717e-02 * humidity**2 + 1.22874e-03 * temperature**2 * humidity + 8.5282e-04 * temperature * humidity**2 - 1.99e-06 * temperature**2 * humidity**2
    wind_chill = 13.12 + 0.6215 * temperature - 11.37 * (wind_speed ** 0.16) + 0.3965 * temperature * (wind_speed ** 0.16)
    
    # One-hot encode categorical features
    cloud_cover_clear = cloud_cover == "clear"
    cloud_cover_cloudy = cloud_cover == "cloudy"
    cloud_cover_overcast = cloud_cover == "overcast"
    cloud_cover_partly_cloudy = cloud_cover == "partly cloudy"
    
    season_autumn = season == "Autumn"
    season_spring = season == "Spring"
    season_summer = season == "Summer"
    season_winter = season == "Winter"
    
    location_coastal = location == "coastal"
    location_inland = location == "inland"
    location_mountain = location == "mountain"
    
    bins = [0, 2, 5, 7, 10, np.inf]
    labels = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    uv_index_binned = pd.cut([uv_index_value], bins=bins, labels=labels)[0]
    
    uv_index_low = uv_index_binned == "Low"
    uv_index_moderate = uv_index_binned == "Moderate"
    uv_index_high = uv_index_binned == "High"
    uv_index_very_high = uv_index_binned == "Very High"
    uv_index_extreme = uv_index_binned == "Extreme"
    
    # Create a button for prediction
    if st.button("Predict"):
        # Prepare the input data
        input_data = np.array([[
            precipitation, atmospheric_pressure, visibility, heat_index, wind_chill,
            cloud_cover_clear, cloud_cover_cloudy, cloud_cover_overcast, cloud_cover_partly_cloudy,
            season_autumn, season_spring, season_summer, season_winter,
            location_coastal, location_inland, location_mountain,
            uv_index_low, uv_index_moderate, uv_index_high, uv_index_very_high, uv_index_extreme
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)

        weather_type=label_to_weather_type[prediction[0]]
        
        # Display the prediction
        st.success(f"The predicted weather type is: {weather_type}")

        # Display the weather type image
        image_path = weather_type_to_image[weather_type]
        st.image(image_path, use_column_width=True)

if __name__ == "__main__":
    main()