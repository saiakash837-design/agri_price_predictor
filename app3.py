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
    except Exception as e:
        st.sidebar.error(f"Price API Error: {e}")
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
    except Exception as e:
        st.sidebar.error(f"Weather API Error: {e}")
    return pd.DataFrame()

# --- STEP 3: MASTER MERGE ---
'''def get_updated_dataset(df_original, df_p, df_w, crop):
    if df_p.empty or df_w.empty:
        return df_original[df_original['Commodity'] == crop]
    
    api_combined = pd.merge(df_p, df_w, on='DATE', how='inner')
    api_combined['Commodity'] = crop
    
    df_combined = pd.concat([df_original, api_combined], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['DATE', 'DISTRICT', 'Commodity']).sort_values('DATE')
    return df_combined

# --- UI & INITIAL LOAD ---
st.sidebar.title("🌿 Nada Control Center")

try:
    df_main = pd.read_csv('Agri_Weather_Lite.csv')
    # Ensure Price Date exists or fallback to DATE
    date_col = 'Price Date' if 'Price Date' in df_main.columns else 'DATE'
    df_main['DATE'] = pd.to_datetime(df_main[date_col])
except Exception as e:
    st.error(f"CSV Load Error: {e}. Please ensure 'Agri_Weather_Lite.csv' is in your directory.")
    st.stop()'''
# --- 1. UPDATE THE MASTER MERGE (Step 3) ---
def get_updated_dataset(df_original, df_p, df_w, crop):
    # Ensure the original dataframe uses 'PRICE' instead of 'Modal_Price'
    if 'Modal_Price' in df_original.columns:
        df_original = df_original.rename(columns={'Modal_Price': 'PRICE'})
    
    if df_p.empty or df_w.empty:
        return df_original[df_original['Commodity'] == crop]
    
    # Standardize API Price column just in case
    df_p = df_p.rename(columns={'modal_price': 'PRICE'})
    
    api_combined = pd.merge(df_p, df_w, on='DATE', how='inner')
    api_combined['Commodity'] = crop
    
    # Combine
    df_combined = pd.concat([df_original, api_combined], ignore_index=True)
    
    # Final Rename check to be 100% safe before returning
    df_combined = df_combined.rename(columns={'Modal_Price': 'PRICE'})
    
    df_combined = df_combined.drop_duplicates(subset=['DATE', 'DISTRICT', 'Commodity']).sort_values('DATE')
    return df_combined

# --- 2. UPDATE THE INITIAL LOAD SECTION ---
try:
    df_main = pd.read_csv('Agri_Weather_Lite.csv')
    
    # CRITICAL FIX: Rename Modal_Price to PRICE immediately after loading
    if 'Modal_Price' in df_main.columns:
        df_main = df_main.rename(columns={'Modal_Price': 'PRICE'})
    
    # Handle Date column
    date_col = 'Price Date' if 'Price Date' in df_main.columns else 'DATE'
    df_main['DATE'] = pd.to_datetime(df_main[date_col])
    
except Exception as e:
    st.error(f"CSV Load Error: {e}")
    st.stop()
crop = st.sidebar.selectbox("Choose Crop", df_main['Commodity'].unique())
location = st.sidebar.text_input("Enter District (e.g., Nagpur)", "Nagpur")
model_choice = st.sidebar.radio("Select Prediction Engine", ["Random Forest", "XGBoost"])

# STARTUP VIEW
if "data_synced" not in st.session_state:
    st.info("Welcome to Nada Harvest AI. Click 'Sync Market & Predict' in the sidebar to begin.")
    st.image("https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=800&q=80", caption="Smart Farming Analysis")

if st.sidebar.button("🔄 Sync Market & Predict"):
    st.session_state.data_synced = True
    
    with st.spinner("Fetching Live Data and Training Models..."):
        # Steps 1, 2, 3
        df_p = fetch_and_clean_prices(crop)
        df_w = fetch_and_clean_weather(location)
        df_updated = get_updated_dataset(df_main, df_p, df_w, crop)
        
        # Filter for selected crop after merge
        df_crop = df_updated[df_updated['Commodity'] == crop].copy()
        
        # Feature Engineering
        df_crop['month'] = df_crop['DATE'].dt.month
        df_crop['day'] = df_crop['DATE'].dt.dayofweek
        df_crop['lag_1'] = df_crop['PRICE'].shift(1)
        df_ml = df_crop.dropna().reset_index(drop=True)
        
        if len(df_ml) < 5:
            st.error("Not enough data to train the model. Try a different crop or location.")
        else:
            features = ['month', 'day', 'TEMP', 'RAIN', 'HUM', 'lag_1']
            X = df_ml[features]
            y = df_ml['PRICE']

            # STEP 4: TRAIN MODELS
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08).fit(X, y)

            # STEP 5: 7-DAY PREDICTION
            last_known = df_ml.iloc[-1]
            forecast = []
            current_lag = last_known['PRICE']
            
            for i in range(1, 8):
                # Using current weather as proxy
                input_data = pd.DataFrame([[last_known['month'], (last_known['day']+i)%7, last_known['TEMP'], 0, last_known['HUM'], current_lag]], columns=features)
                pred = xgb_model.predict(input_data)[0]
                forecast_date = (last_known['DATE'] + timedelta(days=i)).strftime('%Y-%m-%d')
                forecast.append({"Date": forecast_date, "Price (₹)": round(float(pred), 2)})
                current_lag = pred
            
            # --- DISPLAY RESULTS ---
            st.header(f"📊 Market Analysis: {crop} in {location}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = rf_model if model_choice == "Random Forest" else xgb_model
                df_ml['Predicted'] = selected_model.predict(X)
                
                fig = px.line(df_ml, x='DATE', y=['PRICE', 'Predicted'], 
                             title=f"Price History vs {model_choice} Fit",
                             labels={"value": "Price (₹)", "DATE": "Timeline"},
                             color_discrete_map={"PRICE": "#1f77b4", "Predicted": "#ff7f0e"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📅 7-Day Price Forecast")
                st.dataframe(pd.DataFrame(forecast), hide_index=True)
                
                st.divider()
                st.subheader("⚖️ Model Accuracy")
                rf_acc = r2_score(y, rf_model.predict(X))
                xgb_acc = r2_score(y, xgb_model.predict(X))
                
                st.metric("Random Forest R²", f"{rf_acc:.4f}")
                st.metric("XGBoost R²", f"{xgb_acc:.4f}")
                
                if xgb_acc > rf_acc:
                    st.success("XGBoost is the recommended engine.")
                else:
                    st.success("Random Forest is the recommended engine.")

            st.info(f"Analysis based on {len(df_updated)} historical and live data points.")

