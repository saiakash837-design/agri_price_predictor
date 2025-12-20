import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from datetime import timedelta
# --- CONFIGURATION ---
AGMARK_API_KEY = "579b464db66ec23bdd00000153830512e3d048f848bcb6701db55152"
WEATHER_API_KEY = "2N9PMG25KADQ6GE3KMJVHE9XW"

st.set_page_config(page_title="Nada Harvest AI", layout="wide")

# --- STEP 1 & 2: API FETCHING & CLEANING ---
@st.cache_data(ttl=3600)
def fetch_and_clean_prices(commodity):
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={AGMARK_API_KEY}&format=json&filters[commodity]={commodity}&limit=100"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res.get('records', []))
        if not df.empty:
            df = df.rename(columns={'arrival_date': 'DATE', 'modal_price': 'PRICE', 'district': 'DISTRICT'})
            df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
            df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
            return df[['DATE', 'DISTRICT', 'PRICE']]
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_and_clean_weather(location):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res.get('days', []))
        if not df.empty:
            df = df.rename(columns={'datetime': 'DATE', 'temp': 'TEMP', 'precip': 'RAIN', 'humidity': 'HUM'})
            df['DATE'] = pd.to_datetime(df['DATE'])
            return df[['DATE', 'TEMP', 'RAIN', 'HUM']]
    except: pass
    return pd.DataFrame()

# --- STEP 3: MASTER MERGE ---
def get_updated_dataset(df_original, df_p, df_w, crop):
    # Merge API data
    api_combined = pd.merge(df_p, df_w, on='DATE', how='inner')
    api_combined['Commodity'] = crop
    
    # Add to original (Append, don't replace)
    df_combined = pd.concat([df_original, api_combined], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['DATE', 'DISTRICT', 'Commodity']).sort_values('DATE')
    return df_combined

# --- UI & LOGIC ---
st.sidebar.title("🌿 Nada Control Center")
# Replace with your actual CSV name
try:
    df_main = pd.read_csv('Agri_Weather_Lite.csv')
    df_main['DATE'] = pd.to_datetime(df_main['Price Date'])
except:
    st.error("Please ensure 'Agri_Weather_Lite.csv' is in your directory.")
    st.stop()

crop = st.sidebar.selectbox("Choose Crop", df_main['Commodity'].unique())
location = st.sidebar.text_input("Enter District (e.g., Nagpur)", "Nagpur")
model_choice = st.sidebar.radio("Select Prediction Engine", ["Random Forest", "XGBoost"])

if st.sidebar.button("🔄 Sync Market & Predict"):
    # Step 1 & 2
    df_p = fetch_and_clean_prices(crop)
    df_w = fetch_and_clean_weather(location)
    
    # Step 3
    df_updated = get_updated_dataset(df_main[df_main['Commodity']==crop], df_p, df_w, crop)
    
    # Feature Engineering
    df_ml = df_updated.copy()
    df_ml['month'] = df_ml['DATE'].dt.month
    df_ml['day'] = df_ml['DATE'].dt.dayofweek
    df_ml['lag_1'] = df_ml['PRICE'].shift(1)
    df_ml = df_ml.dropna()
    
    features = ['month', 'day', 'TEMP', 'RAIN', 'HUM', 'lag_1']
    X = df_ml[features]
    y = df_ml['PRICE']

    # STEP 4: TRAIN MODELS
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08).fit(X, y)

    # STEP 5: 7-DAY PREDICTION (XGBOOST)
    last_known = df_ml.iloc[-1]
    forecast = []
    current_lag = last_known['PRICE']
    
    for i in range(1, 8):
        # We use today's weather as a proxy for the forecast week
        pred = xgb_model.predict([[last_known['month'], (last_known['day']+i)%7, last_known['TEMP'], 0, last_known['HUM'], current_lag]])[0]
        forecast.append({"Day": (last_known['DATE'] + timedelta(days=i)).strftime('%Y-%m-%d'), "Forecasted Price": round(pred, 2)})
        current_lag = pred
    
    df_forecast = pd.DataFrame(forecast)

    # --- DISPLAY RESULTS ---
    st.header(f"📊 Analysis: {crop} in {location}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected = rf_model if model_choice == "Random Forest" else xgb_model
        df_ml['Predicted'] = selected.predict(X)
        fig = px.line(df_ml, x='DATE', y=['PRICE', 'Predicted'], 
                     title=f"Historical Price vs {model_choice} Fit",
                     color_discrete_map={"PRICE": "#1f77b4", "Predicted": "#ff7f0e"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 7-Day Forecast")
        st.table(df_forecast)
        
        if st.button("⚖️ Compare Model Accuracy"):
            rf_acc = r2_score(y, rf_model.predict(X))
            xgb_acc = r2_score(y, xgb_model.predict(X))
            st.write(f"**Random Forest Accuracy (R²):** `{rf_acc:.4f}`")
            st.write(f"**XGBoost Accuracy (R²):** `{xgb_acc:.4f}`")
            best = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
            st.success(f"Winner: {best}")

    # Step 3 visualization: Show that data was added

    st.info(f"Total data points analyzed: {len(df_updated)} (Updated with live API data)")

