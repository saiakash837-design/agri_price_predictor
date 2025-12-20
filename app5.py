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

# --- CONFIGURATION ---
AGMARK_API_KEY = "579b464db66ec23bdd00000153830512e3d048f848bcb6701db55152"

# Static Coordinates for the Map
CITY_COORDS = {
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Nashik": {"lat": 20.0050, "lon": 73.7889},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
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

    # UPDATED: Added Ensemble option
    model_choice = st.sidebar.radio("AI Engine", ["Random Forest", "XGBoost", "LightGBM", "Ensemble (Consensus)"])
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

def display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, current_price):
    st.write("---")
    st.subheader(f"🌐 Regional Prices for {selected_commodity} in {selected_state}")
    
    map_df = df_base[(df_base['STATE'] == selected_state) & (df_base['COMMODITY'] == selected_commodity)].copy()
    latest_date = map_df['DATE'].max()
    map_df = map_df[map_df['DATE'] == latest_date]

    if map_df.empty:
        st.warning("No nearby market data found.")
        return

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
            st.success(f"🚀 **Arbitrage Alert:** Potential profit of **₹{diff:.2f}/quintal** in **{top_m['MARKET']}**!")

# --- 4. MAIN ACTION ---
if st.sidebar.button("🔄 Sync & Predict"):
    df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
    df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
    
    df_master = pd.concat([df_hist, df_live], ignore_index=True).drop_duplicates('DATE').sort_values('DATE')
    
    # Feature Engineering
    df_master['month'] = df_master['DATE'].dt.month
    df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
    df_master['lag_1'] = df_master['PRICE'].shift(1)
    df_master['lag_7'] = df_master['PRICE'].shift(7)
    df_ml = df_master.dropna()

    if len(df_ml) > 10:
        # Standard ML Training
        features = ['month', 'dayofweek', 'lag_1', 'lag_7']
        X, y = df_ml[features], df_ml['PRICE']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)
        lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1).fit(X, y)

        # Prophet Specific Training
        df_p = df_ml[['DATE', 'PRICE']].rename(columns={'DATE': 'ds', 'PRICE': 'y'})
        prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
        prophet_model.fit(df_p)

        # --- PREDICTION LOGIC ---
        last_row = df_ml.iloc[-1]
        forecast = []
        curr_price_rf = curr_price_xgb = curr_price_lgb = last_row['PRICE']
        
        # Prophet Forecast for 7 days
        future = prophet_model.make_future_dataframe(periods=7)
        prophet_forecast = prophet_model.predict(future).tail(7)['yhat'].values

        for i in range(1, 8):
            f_date = last_row['DATE'] + timedelta(days=i)
            in_df = pd.DataFrame([[f_date.month, f_date.dayofweek, 0, last_row['lag_7']]], columns=features)
            
            # Update lag_1 for recursive prediction
            in_df['lag_1'] = curr_price_rf
            p_rf = rf.predict(in_df)[0]
            
            in_df['lag_1'] = curr_price_xgb
            p_xgb = xgb.predict(in_df)[0]
            
            in_df['lag_1'] = curr_price_lgb
            p_lgb = lgb.predict(in_df)[0]
            
            p_prophet = prophet_forecast[i-1]

            # Ensemble Averaging
            p_ensemble = (p_rf + p_xgb + p_lgb + p_prophet) / 4
            
            # Select which prediction to show based on sidebar
            if model_choice == "Random Forest": selected_p = p_rf
            elif model_choice == "XGBoost": selected_p = p_xgb
            elif model_choice == "LightGBM": selected_p = p_lgb
            else: selected_p = p_ensemble
            
            forecast.append({"Date": f_date.strftime('%Y-%m-%d'), "Price": round(float(selected_p), 2)})
            curr_price_rf, curr_price_xgb, curr_price_lgb = p_rf, p_xgb, p_lgb

        # --- DISPLAY RESULTS ---
        st.title(f"🌾 {selected_commodity} Intelligence")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Historical Accuracy Line
            df_ml['RF'] = rf.predict(X)
            df_ml['XGB'] = xgb.predict(X)
            df_ml['LGB'] = lgb.predict(X)
            # Simplified Ensemble for historical plot
            df_ml['Ensemble'] = (df_ml['RF'] + df_ml['XGB'] + df_ml['LGB']) / 3
            
            plot_col = "Ensemble" if "Ensemble" in model_choice else model_choice.split()[0]
            if plot_col == "LightGBM": plot_col = "LGB"
            
            fig = px.line(df_ml, x='DATE', y=['PRICE', plot_col], title=f"Model Performance: {model_choice}")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("📅 7-Day Forecast")
            st.table(pd.DataFrame(forecast))
            
            score = r2_score(y, df_ml[plot_col])
            st.metric("Model Confidence (R²)", f"{score:.3f}")

        display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, last_row['PRICE'])

    else:
        st.warning("Insufficient data points for AI training. Please select a different market.")
