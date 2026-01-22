import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import r2_score
from datetime import timedelta
import os

# --- CONFIGURATION & SETTINGS ---
st.set_page_config(page_title="Nada Harvest AI", layout="wide", page_icon="🌾")

AGMARK_API_KEY = "579b464db66ec23bdd00000153830512e3d048f848bcb6701db55152"

CITY_COORDS = {
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Nashik": {"lat": 20.0050, "lon": 73.7889},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
}

# --- STYLING & HERO SECTION ---
def show_hero():
    st.markdown("""
        <div style="background-color:#1E3A1E;padding:40px;border-radius:15px;text-align:center;margin-bottom:25px">
            <h1 style="color:#FFFFFF;margin-bottom:0">🌾 NADA HARVEST AI</h1>
            <p style="color:#A8E6A1;font-size:20px;margin-top:10px">Advanced Agricultural Intelligence: Price Forecasting & Arbitrage Detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Market Sync", "Active", "Government API")
    col2.metric("AI Models", "4 Engines", "Ensemble Active")
    col3.metric("Coverage", "National", "All States")

@st.cache_data
def load_base_data():
    try:
        df = pd.read_csv('Agri_Weather_Lite.csv')
        name_map = {
            'Price Date': 'DATE', 
            'Modal_Price': 'PRICE', 
            'Market Name': 'MARKET', 
            'STATE': 'STATE', 
            'Commodity': 'COMMODITY',
            'Temparature': 'Temperature' 
        }
        df = df.rename(columns=name_map)
        df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        return df.dropna(subset=['PRICE'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# --- 1. DATA LOADING ---
df_base = load_base_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("📍 Market Settings")
if not df_base.empty:
    states = sorted(df_base['STATE'].unique())
    selected_state = st.sidebar.selectbox("Choose State", states)
    state_df = df_base[df_base['STATE'] == selected_state]
    markets = sorted(state_df['MARKET'].unique())
    selected_market = st.sidebar.selectbox("Choose Market", markets)
    market_df = state_df[state_df['MARKET'] == selected_market]
    commodities = sorted(market_df['COMMODITY'].unique())
    selected_commodity = st.sidebar.selectbox("Choose Commodity", commodities)

    model_choice = st.sidebar.radio("Select AI Engine", 
                                   ["Random Forest", "XGBoost", "LightGBM", "Prophet", "Ensemble (Consensus)"])
else:
    st.error("Missing dataset. Please upload Agri_Weather_Lite.csv")
    st.stop()

# --- 3. CORE LOGIC ---
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

def display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, current_price):
    st.write("---")
    st.subheader(f"🌐 Regional Opportunity Map: {selected_commodity}")
    map_df = df_base[(df_base['STATE'] == selected_state) & (df_base['COMMODITY'] == selected_commodity)].copy()
    latest_date = map_df['DATE'].max()
    map_df = map_df[map_df['DATE'] == latest_date]
    
    if not map_df.empty:
        def get_coords(row, key):
            data = CITY_COORDS.get(row['MARKET'], CITY_COORDS.get(row['STATE'], {"lat": 20.5937, "lon": 78.9629}))
            return data[key]
        map_df['lat'] = map_df.apply(lambda r: get_coords(r, 'lat'), axis=1) + np.random.uniform(-0.1, 0.1, len(map_df))
        map_df['lon'] = map_df.apply(lambda r: get_coords(r, 'lon'), axis=1) + np.random.uniform(-0.1, 0.1, len(map_df))
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="PRICE", color="PRICE",
                                color_continuous_scale="YlOrRd", hover_name="MARKET", zoom=5, mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        top_m = map_df.sort_values('PRICE', ascending=False).iloc[0]
        if top_m['MARKET'] != selected_market:
            diff = top_m['PRICE'] - current_price
            if diff > 0:
                st.success(f"🚀 **Arbitrage Opportunity:** Earn **₹{diff:.2f}/quintal** more in **{top_m['MARKET']}**!")

# --- 4. EXECUTION ---
show_hero()

if st.sidebar.button("🔄 Sync & Predict"):
    with st.spinner("Synchronizing with Government API and training AI models..."):
        df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
        df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
        df_master = pd.concat([df_hist, df_live], ignore_index=True).drop_duplicates('DATE').sort_values('DATE')
        
        # FEATURE ENGINEERING
        df_master['month'] = df_master['DATE'].dt.month
        df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
        df_master['lag_1'] = df_master['PRICE'].shift(1)
        df_master['lag_7'] = df_master['PRICE'].shift(7)
        df_master['Comm_Code'] = df_master['COMMODITY'].astype('category').cat.codes
        
        # --- DYNAMIC FEATURE HANDLING ---
        potential_features = ['month', 'dayofweek', 'lag_1', 'lag_7', 'Humidity', 'Temperature', 'Rainfall', 'Comm_Code']
        # Only use columns that exist in the CSV/Dataframe
        existing_features = [f for f in potential_features if f in df_master.columns]
        # Columns that MUST be non-null for the ML model to train
        dropna_subset = [f for f in ['PRICE', 'lag_1', 'lag_7'] if f in existing_features] + ['PRICE']
        
        df_ml = df_master.dropna(subset=dropna_subset)

        if len(df_ml) > 12:
            X, y = df_ml[existing_features], df_ml['PRICE']
            
            # Model Training
            rf = RandomForestRegressor(n_estimators=100).fit(X, y)
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)
            lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1).fit(X, y)
            
            df_p = df_ml[['DATE', 'PRICE']].rename(columns={'DATE': 'ds', 'PRICE': 'y'})
            prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(df_p)

            # Forecasting 7 Days
            last_row = df_ml.iloc[-1]
            forecast_data = []
            c_rf, c_xgb, c_lgb = last_row['PRICE'], last_row['PRICE'], last_row['PRICE']
            
            p_future = prophet_model
