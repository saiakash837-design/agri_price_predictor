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

st.set_page_config(page_title="Nada Harvest: Price Predictor", layout="wide")

# --- STEP 1 & 2: API FETCHING & CLEANING ---
@st.cache_data(ttl=3600)
def fetch_live_prices(commodity, state, market):
    # Filtering API by commodity, state, and market to keep data light
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={AGMARK_API_KEY}&format=json&filters[commodity]={commodity}&filters[state]={state}&filters[market]={market}&limit=50"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res.get('records', []))
        if not df.empty:
            df = df.rename(columns={'arrival_date': 'DATE', 'modal_price': 'PRICE', 'state': 'STATE', 'market': 'MARKET'})
            df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
            df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
            return df[['DATE', 'STATE', 'MARKET', 'PRICE']]
    except: pass
    return pd.DataFrame()

# --- INITIAL DATA LOAD ---
@st.cache_data
def load_local_data():
    try:
        df = pd.read_csv('Agri_Weather_Lite.csv')
        # Standardize Columns
        rename_map = {'Price Date': 'DATE', 'Modal_Price': 'PRICE', 'Market Name': 'MARKET'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df_base = load_local_data()

if df_base.empty:
    st.stop()

# --- STEP 3: DYNAMIC UI (NO TYPING) ---
st.sidebar.title("🌿 Price Control Center")

# Dynamic Dropdowns
states = sorted(df_base['STATE'].unique())
selected_state = st.sidebar.selectbox("Select State", states)

filtered_by_state = df_base[df_base['STATE'] == selected_state]
commodities = sorted(filtered_by_state['Commodity'].unique())
selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)

filtered_by_comm = filtered_by_state[filtered_by_state['Commodity'] == selected_commodity]
markets = sorted(filtered_by_comm['MARKET'].unique())
selected_market = st.sidebar.selectbox("Select Market", markets)

model_choice = st.sidebar.radio("ML Engine", ["Random Forest", "XGBoost"])

# --- PROCESS & PREDICT ---
if st.sidebar.button("🔄 Sync Live Data & Predict"):
    # 1. Get Live Data
    df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
    
    # 2. Filter Historical Data for the specific selection
    df_hist = filtered_by_comm[filtered_by_comm['MARKET'] == selected_market].copy()
    
    # 3. Master Merge (Combine Hist + Live)
    df_final = pd.concat([df_hist, df_live], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['DATE']).sort_values('DATE')
    
    # 4. Feature Engineering (Time-based features)
    df_final['month'] = df_final['DATE'].dt.month
    df_final['day_of_week'] = df_final['DATE'].dt.dayofweek
    df_final['lag_1'] = df_final['PRICE'].shift(1)
    df_final['lag_7'] = df_final['PRICE'].shift(7)
    
    df_ml = df_final.dropna()

    if len(df_ml) < 10:
        st.warning("Insufficient historical data for this specific market to run AI models.")
    else:
        # ML Training
        features = ['month', 'day_of_week', 'lag_1', 'lag_7']
        X = df_ml[features]
        y = df_ml['PRICE']

        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)

        # 5. 7-Day Forecast (XGBoost)
        last_row = df_ml.iloc[-
