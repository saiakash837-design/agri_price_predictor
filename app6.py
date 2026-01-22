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
            'Price Date': 'DATE', 'Modal_Price': 'PRICE', 
            'Market Name': 'MARKET', 'STATE': 'STATE', 
            'Commodity': 'COMMODITY', 'Temparature': 'Temperature'
        }
        df = df.rename(columns=name_map)
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['PRICE', 'DATE'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df_base = load_base_data()

# --- SIDEBAR ---
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
    model_choice = st.sidebar.radio("Select AI Engine", ["Random Forest", "XGBoost", "LightGBM", "Prophet", "Ensemble (Consensus)"])
else:
    st.stop()

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
    if not map_df.empty:
        latest_date = map_df['DATE'].max()
        map_df = map_df[map_df['DATE'] == latest_date]
        def get_coords(row, key):
            data = CITY_COORDS.get(row['MARKET'], CITY_COORDS.get(row['STATE'], {"lat": 20.5937, "lon": 78.9629}))
            return data[key]
        map_df['lat'] = map_df.apply(lambda r: get_coords(r, 'lat'), axis=1) + np.random.uniform(-0.1, 0.1, len(map_df))
        map_df['lon'] = map_df.apply(lambda r: get_coords(r, 'lon'), axis=1) + np.random.uniform(-0.1, 0.1, len(map_df))
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="PRICE", color="PRICE", color_continuous_scale="YlOrRd", zoom=5, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
        top_m = map_df.sort_values('PRICE', ascending=False).iloc[0]
        if top_m['MARKET'] != selected_market and top_m['PRICE'] > current_price:
            st.success(f"🚀 **Arbitrage Opportunity:** Earn **₹{top_m['PRICE'] - current_price:.2f}/quintal** more in **{top_m['MARKET']}**!")

show_hero()

if st.sidebar.button("🔄 Sync & Predict"):
    with st.spinner("Analyzing Market Data..."):
        df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
        df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
        df_master = pd.concat([df_hist, df_live], ignore_index=True).drop_duplicates('DATE').sort_values('DATE')
        
        # Feature Engineering
        df_master['month'] = df_master['DATE'].dt.month
        df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
        df_master['lag_1'] = df_master['PRICE'].shift(1)
        df_master['lag_7'] = df_master['PRICE'].shift(7)
        df_master['Comm_Code'] = df_master['COMMODITY'].astype('category').cat.codes
        
        # --- DYNAMIC FEATURE FIX ---
        potential_features = ['month', 'dayofweek', 'lag_1', 'lag_7', 'Humidity', 'Temperature', 'Rainfall', 'Comm_Code']
        # Only use features that actually exist in your CSV
        actual_features = [f for f in potential_features if f in df_master.columns]
        
        df_ml = df_master.dropna(subset=['PRICE', 'lag_1', 'lag_7'])

        if len(df_ml) > 10:
            X, y = df_ml[actual_features], df_ml['PRICE']
            
            # Models
            rf = RandomForestRegressor(n_estimators=100).fit(X, y)
            xgb = XGBRegressor(n_estimators=100).fit(X, y)
            lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
            
            # Prophet
            df_p = df_ml[['DATE', 'PRICE']].rename(columns={'DATE': 'ds', 'PRICE': 'y'})
            prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(df_p)

            # 7-Day Forecast
            last_row = df_ml.iloc[-1]
            forecast_data = []
            c_rf = c_xgb = c_lgb = last_row['PRICE']
            p_preds = prophet_model.predict(prophet_model.make_future_dataframe(periods=7)).tail(7)['yhat'].values

            for i in range(1, 8):
                f_date = last_row['DATE'] + timedelta(days=i)
                # Build input for this specific date
                in_dict = {f: last_row[f] for f in actual_features if f not in ['month', 'dayofweek', 'lag_1']}
                in_dict.update({'month': f_date.month, 'dayofweek': f_date.dayofweek})
                
                # Model-specific lag updates
                temp_df = pd.DataFrame([in_dict])
                temp_df['lag_1'] = c_rf
                p1 = rf.predict(temp_df[actual_features])[0]
                temp_df['lag_1'] = c_xgb
                p2 = xgb.predict(temp_df[actual_features])[0]
                temp_df['lag_1'] = c_lgb
                p3 = lgb.predict(temp_df[actual_features])[0]
                p4 = p_preds[i-1]
                
                ens = (p1 + p2 + p3 + p4) / 4
                mapping = {"Random Forest": p1, "XGBoost": p2, "LightGBM": p3, "Prophet": p4, "Ensemble (Consensus)": ens}
                forecast_data.append({"Date": f_date.strftime('%Y-%m-%d'), "Price": round(mapping[model_choice], 2)})
                c_rf, c_xgb, c_lgb = p1, p2, p3

            # --- DISPLAY RESULTS ---
            st.title(f"📊 {selected_commodity} Insights")
            col_acc, col_table = st.columns([2, 1])
            
            with col_acc:
                df_ml['RF'], df_ml['XGB'], df_ml['LGB'] = rf.predict(X), xgb.predict(X), lgb.predict(X)
                df_ml['Prophet'] = prophet_model.predict(df_p)['yhat'].values
                df_ml['Ensemble'] = (df_ml['RF'] + df_ml['XGB'] + df_ml['LGB'] + df_ml['Prophet']) / 4
                
                col_map = {"Random Forest": "RF", "XGBoost": "XGB", "LightGBM": "LGB", "Prophet": "Prophet", "Ensemble (Consensus)": "Ensemble"}
                fig = px.line(df_ml, x='DATE', y=['PRICE', col_map[model_choice]], title=f"Prediction vs Actual: {model_choice}")
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.subheader("📅 7-Day Prediction")
                st.table(pd.DataFrame(forecast_data))
                st.metric("Model Confidence (R²)", f"{r2_score(y, df_ml[col_map[model_choice]]):.3f}")

            display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, last_row['PRICE'])
        else:
            st.warning("Not enough data to generate a forecast for this selection.")
else:
    st.info("👈 Select parameters and click **Sync & Predict**.")
