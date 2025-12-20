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

st.set_page_config(page_title="Nada Harvest AI", layout="wide")

# --- 1. DATA LOADING & CLEANING ---
@st.cache_data
def load_base_data():
    try:
        # Load your historical CSV
        df = pd.read_csv('Agri_Weather_Lite.csv')
        
        # Standardize column names immediately
        name_map = {
            'Price Date': 'DATE', 
            'Modal_Price': 'PRICE', 
            'Market Name': 'MARKET',
            'STATE': 'STATE',
            'Commodity': 'COMMODITY'
        }
        df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df.dropna(subset=['PRICE'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df_base = load_base_data()

# --- 2. CASCADING SIDEBAR FILTERS ---
st.sidebar.header("📍 Market Selection")

if not df_base.empty:
    # State Selection
    states = sorted(df_base['STATE'].unique())
    selected_state = st.sidebar.selectbox("Choose State", states)

    # Filter by State -> Market Selection
    state_df = df_base[df_base['STATE'] == selected_state]
    markets = sorted(state_df['MARKET'].unique())
    selected_market = st.sidebar.selectbox("Choose Market", markets)

    # Filter by Market -> Commodity Selection
    market_df = state_df[state_df['MARKET'] == selected_market]
    commodities = sorted(market_df['COMMODITY'].unique())
    selected_commodity = st.sidebar.selectbox("Choose Commodity", commodities)

    model_choice = st.sidebar.radio("AI Engine", ["Random Forest", "XGBoost"])
else:
    st.error("Dataset not found. Please check your file paths.")
    st.stop()

# --- 3. SYNC FUNCTION (API FETCH) ---
@st.cache_data(ttl=3600)
def fetch_live_prices(comm, state, market):
    # Standardize names for the API (replacing spaces with %20)
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={AGMARK_API_KEY}&format=json&filters[commodity]={comm}&filters[state]={state}&filters[market]={market}&limit=50"
    try:
        res = requests.get(url).json()
        df_api = pd.DataFrame(res.get('records', []))
        if not df_api.empty:
            df_api = df_api.rename(columns={'arrival_date': 'DATE', 'modal_price': 'PRICE'})
            df_api['DATE'] = pd.to_datetime(df_api['DATE'], dayfirst=True)
            df_api['PRICE'] = pd.to_numeric(df_api['PRICE'], errors='coerce')
            return df_api[['DATE', 'PRICE']]
    except:
        pass
    return pd.DataFrame()

# --- 4. CORE LOGIC ---
if st.sidebar.button("🔄 Sync & Predict"):
    # Filter historical data
    df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
    
    # Fetch Live Data
    df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
    
    # Step 3: Master Merge (Append Live to Hist)
    df_master = pd.concat([df_hist, df_live], ignore_index=True)
    df_master = df_master.drop_duplicates(subset=['DATE']).sort_values('DATE')
    
    # Feature Engineering
    df_master['month'] = df_master['DATE'].dt.month
    df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
    df_master['lag_1'] = df_master['PRICE'].shift(1)
    df_master['lag_7'] = df_master['PRICE'].shift(7)
    
    df_ml = df_master.dropna()

    if len(df_ml) > 7:
        # STEP 4: TRAINING
        features = ['month', 'dayofweek', 'lag_1', 'lag_7']
        X = df_ml[features]
        y = df_ml['PRICE']

        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)

        # STEP 5: 7-DAY FORECAST
        last_row = df_ml.iloc[-1]
        forecast = []
        curr_price = last_row['PRICE']
        
        for i in range(1, 8):
            f_date = last_row['DATE'] + timedelta(days=i)
            # Simplified forecast using price momentum
            input_df = pd.DataFrame([[f_date.month, f_date.dayofweek, curr_price, last_row['lag_7']]], columns=features)
            pred = xgb.predict(input_df)[0]
            forecast.append({"Date": f_date.strftime('%Y-%m-%d'), "Forecasted Price": round(float(pred), 2)})
            curr_price = pred
        
        df_forecast = pd.DataFrame(forecast)

        # --- STEP 6: DISPLAY ---
        st.title(f"🌾 {selected_commodity} in {selected_market}")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            active_model = rf if model_choice == "Random Forest" else xgb
            df_ml['Predicted'] = active_model.predict(X)
            fig = px.line(df_ml, x='DATE', y=['PRICE', 'Predicted'], 
                          title="Price Trend Analysis",
                          labels={"value": "Price (₹)", "variable": "Legend"})
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("📅 7-Day Prediction")
            st.table(df_forecast)
            
            # Accuracy Metrics
            rf_score = r2_score(y, rf.predict(X))
            xgb_score = r2_score(y, xgb.predict(X))
            st.metric("Random Forest R²", f"{rf_score:.3f}")
            st.metric("XGBoost R²", f"{xgb_score:.3f}")

    else:
        st.warning(f"Insufficient historical data ({len(df_ml)} rows) to train the model for this market. Try a larger market or a different commodity.")

else:
    # Default Welcome Screen
    st.title("🌾 Nada Harvest: Agri Intelligence")
    st.info("Select a State and Market from the sidebar, then click **Sync & Predict** to view the AI analysis.")
