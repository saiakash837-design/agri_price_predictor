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

# Static Coordinates for the Map (Hackathon Ready)
CITY_COORDS = {
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nashik": {"lat": 20.0050, "lon": 73.7889},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
}

st.set_page_config(page_title="Nada Harvest AI", layout="wide")

# --- 1. DATA LOADING ---
@st.cache_data
def load_base_data():
    try:
        df = pd.read_csv('Agri_Weather_Lite.csv')
        name_map = {
            'Price Date': 'DATE', 'Modal_Price': 'PRICE', 
            'Market Name': 'MARKET', 'STATE': 'STATE', 'Commodity': 'COMMODITY'
        }
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df.dropna(subset=['PRICE'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df_base = load_base_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("📍 Market Selection")
if not df_base.empty:
    states = sorted(df_base['STATE'].unique())
    selected_state = st.sidebar.selectbox("Choose State", states)

    state_df = df_base[df_base['STATE'] == selected_state]
    markets = sorted(state_df['MARKET'].unique())
    selected_market = st.sidebar.selectbox("Choose Market", markets)

    market_df = state_df[state_df['MARKET'] == selected_market]
    commodities = sorted(market_df['COMMODITY'].unique())
    selected_commodity = st.sidebar.selectbox("Choose Commodity", commodities)

    model_choice = st.sidebar.radio("AI Engine", ["Random Forest", "XGBoost"])
else:
    st.stop()

# --- 3. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_live_prices(comm, state, market):
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={AGMARK_API_KEY}&format=json&filters[commodity]={comm}&filters[state]={state}&filters[market]={market}&limit=50"
    try:
        res = requests.get(url).json()
        df_api = pd.DataFrame(res.get('records', []))
        if not df_api.empty:
            df_api = df_api.rename(columns={'arrival_date': 'DATE', 'modal_price': 'PRICE'})
            df_api['DATE'] = pd.to_datetime(df_api['DATE'], dayfirst=True)
            df_api['PRICE'] = pd.to_numeric(df_api['PRICE'], errors='coerce')
            return df_api[['DATE', 'PRICE']]
    except: pass
    return pd.DataFrame()

def display_map_and_arbitrage(df_all):
    st.write("---")
    st.subheader("🌐 Regional Price Heatmap")
    
    # Map Processing
    latest_date = df_all['DATE'].max()
    map_df = df_all[df_all['DATE'] == latest_date].copy()
    
    def get_coords(row, key):
        data = CITY_COORDS.get(row['MARKET'], CITY_COORDS.get(row['STATE'], {"lat": 20, "lon":
