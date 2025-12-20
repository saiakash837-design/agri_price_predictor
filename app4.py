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
   # --- MAHARASHTRA ---
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Nashik": {"lat": 20.0050, "lon": 73.7889},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433},
    "Solapur": {"lat": 17.6599, "lon": 75.9064},

    # --- KARNATAKA ---
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Hubli": {"lat": 15.3647, "lon": 75.1240},
    "Belgaum": {"lat": 15.8497, "lon": 74.4977},
    "Gulbarga": {"lat": 17.3297, "lon": 76.8343},

    # --- UTTAR PRADESH ---
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Meerut": {"lat": 28.9845, "lon": 77.7064},
    "Bareilly": {"lat": 28.3670, "lon": 79.4304},

    # --- PUNJAB & HARYANA ---
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Haryana": {"lat": 29.0588, "lon": 76.0856},
    "Karnal": {"lat": 29.6857, "lon": 76.9907},

    # --- GUJARAT ---
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022},

    # --- MADHYA PRADESH ---
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Gwalior": {"lat": 26.2183, "lon": 78.1828},

    # --- RAJASTHAN ---
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Kota": {"lat": 25.2138, "lon": 75.8648},

    # --- ANDHRA PRADESH & TELANGANA ---
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Vijayawada": {"lat": 16.5062, "lon": 80.6480},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
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
def display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, current_price):
    st.write("---")
    st.subheader(f"🌐 Regional Prices for {selected_commodity} in {selected_state}")
    
    # 1. CRITICAL CHANGE: Filter for ALL markets in the state for the chosen commodity
    # This ensures multiple dots appear on the map
    map_df = df_base[
        (df_base['STATE'] == selected_state) & 
        (df_base['COMMODITY'] == selected_commodity)
    ].copy()
    
    # Get the most recent date available in this state
    latest_date = map_df['DATE'].max()
    map_df = map_df[map_df['DATE'] == latest_date]

    if map_df.empty:
        st.warning("No nearby market data found for this commodity today.")
        return

    # 2. COORDINATE ASSIGNMENT
    def get_coords(row, key):
    # 1. Try to find the specific Market Name
        if row['MARKET'] in CITY_COORDS:
            return CITY_COORDS[row['MARKET']][key]
    # 2. If Market not found, use the State center
        if row['STATE'] in CITY_COORDS:
            return CITY_COORDS[row['STATE']][key]
    # 3. Ultimate fallback: Center of India
        return 20.5937 if key == 'lat' else 78.9629

    map_df['lat'] = map_df.apply(lambda r: get_coords(r, 'lat'), axis=1) 
    map_df['lon'] = map_df.apply(lambda r: get_coords(r, 'lon'), axis=1)
    
    # Add "Jitter" so multiple markets in one city don't stack perfectly
    map_df['lat'] += np.random.uniform(-0.15, 0.15, len(map_df))
    map_df['lon'] += np.random.uniform(-0.15, 0.15, len(map_df))

    # 3. PLOT ALL MARKETS
    fig = px.scatter_mapbox(
        map_df, 
        lat="lat", 
        lon="lon", 
        size="PRICE", 
        color="PRICE",
        color_continuous_scale=px.colors.sequential.YlOrRd,
        hover_name="MARKET", 
        hover_data={"PRICE": True, "lat": False, "lon": False},
        zoom=5, 
        mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # 4. ARBITRAGE CALCULATION
    top_market = map_df.sort_values('PRICE', ascending=False).iloc[0]
    
    if top_market['MARKET'] != selected_market:
        diff = top_market['PRICE'] - current_price
        if diff > 0:
            st.success(f"🚀 **Arbitrage Alert:** Potential profit of **₹{diff:.2f}/quintal** found in **{top_market['MARKET']}**!")
# --- 4. MAIN ACTION ---
if st.sidebar.button("🔄 Sync & Predict"):
    df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
    df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
    
    df_master = pd.concat([df_hist, df_live], ignore_index=True)
    df_master = df_master.drop_duplicates(subset=['DATE']).sort_values('DATE')
    
    # Feature Engineering
    df_master['month'] = df_master['DATE'].dt.month
    df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
    df_master['lag_1'] = df_master['PRICE'].shift(1)
    df_master['lag_7'] = df_master['PRICE'].shift(7)
    df_ml = df_master.dropna()

    if len(df_ml) > 7:
        # ML Training
        features = ['month', 'dayofweek', 'lag_1', 'lag_7']
        X, y = df_ml[features], df_ml['PRICE']
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)

        # 7-Day Forecast
        last_row = df_ml.iloc[-1]
        forecast = []
        curr_price = last_row['PRICE']
        for i in range(1, 8):
            f_date = last_row['DATE'] + timedelta(days=i)
            input_df = pd.DataFrame([[f_date.month, f_date.dayofweek, curr_price, last_row['lag_7']]], columns=features)
            pred = xgb.predict(input_df)[0]
            forecast.append({"Date": f_date.strftime('%Y-%m-%d'), "Price": round(float(pred), 2)})
            curr_price = pred
        
        # Display Results
        st.title(f"🌾 {selected_commodity} Analysis")
        c1, c2 = st.columns([2, 1])
        with c1:
            active_model = rf if model_choice == "Random Forest" else xgb
            df_ml['Predicted'] = active_model.predict(X)
            st.plotly_chart(px.line(df_ml, x='DATE', y=['PRICE', 'Predicted'], title="Price Trend"), use_container_width=True)
        with c2:
            display_map_and_arbitrage(
            df_base, 
            selected_state, 
            selected_commodity, 
            selected_market, 
            curr_price)
            st.subheader("📅 7-Day Prediction")
            st.table(pd.DataFrame(forecast))
            st.metric("Model Accuracy (R²)", f"{r2_score(y, active_model.predict(X)):.3f}")

        # Integrated Map Feature
       
    else:
        st.warning("Insufficient data for AI training.")
else:
    st.title("🌾 Nada Harvest: Agri Intelligence")
    st.info("Select options in the sidebar and click Sync.")
