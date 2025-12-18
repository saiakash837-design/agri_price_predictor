import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriPrice AI Predictor", layout="wide")
st.title("🌾 Agricultural Price Prediction & Weather Simulation")
st.markdown("Use the sidebar to select a market and simulate how extreme weather affects prices.")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Ensure this matches the filename you uploaded to GitHub
    df = pd.read_csv('Agri_Weather_Lite.csv') 
    df['Price Date'] = pd.to_datetime(df['Price Date'])
    
    # Ensure weather columns are numeric
    for col in ['Temperature_Avg (°C)', 'Rainfall (mm)', 'Humidity (%)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['Modal_Price', 'Temperature_Avg (°C)'])

df = load_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("📍 Market Selection")
state = st.sidebar.selectbox("Choose State", sorted(df['STATE'].unique()))
commodity = st.sidebar.selectbox("Choose Commodity", sorted(df[df['STATE'] == state]['Commodity'].unique()))

df_filtered = df[(df['STATE'] == state) & (df['Commodity'] == commodity)].copy()
market = st.sidebar.selectbox("Choose Market", sorted(df_filtered['Market Name'].unique()))

# Prepare Market-Specific Data
df_market = df_filtered[df_filtered['Market Name'] == market].sort_values('Price Date')
df_market['month'] = df_market['Price Date'].dt.month
df_market['dayofweek'] = df_market['Price Date'].dt.dayofweek
df_market['price_lag_1'] = df_market['Modal_Price'].shift(1)
df_market['price_lag_7'] = df_market['Modal_Price'].shift(7)
df_final = df_market.dropna()

# --- 3. WEATHER SIMULATION SLIDERS (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Weather 'What-If' Simulation")

if not df_final.empty:
    latest_row = df_final.iloc[[-1]]
    curr_temp = float(latest_row['Temperature_Avg (°C)'].values[0])
