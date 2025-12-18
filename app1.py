import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriPrice AI", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Update this filename if yours is different!
        df = pd.read_csv('Agri_Weather_Lite.csv') 
        df['Price Date'] = pd.to_datetime(df['Price Date'])
        
        # Clean numeric columns
        cols = ['Modal_Price', 'Temperature_Avg (°C)', 'Rainfall (mm)', 'Humidity (%)']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Modal_Price'])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("The dataset is empty or the file was not found. Please check your GitHub files.")
    st.stop()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("📍 Market Selection")
all_states = sorted(df['STATE'].unique())
state = st.sidebar.selectbox("Choose State", all_states)

# Filter commodities for the selected state
state_df = df[df['STATE'] == state]
all_commodities = sorted(state_df['Commodity'].unique())
commodity = st.sidebar.selectbox("Choose Commodity", all_commodities)

# Filter markets for the selected state and commodity
comm_df = state_df[state_df['Commodity'] == commodity]
all_markets = sorted(comm_df['Market Name'].unique())
market = st.sidebar.selectbox("Choose Market", all_markets)

# FINAL FILTERED DATA
df_market = comm_df[comm_df['Market Name'] == market].sort_values('Price Date')

# --- 3. DEBUG INFO (Remove this later) ---
with st.expander("🔍 Debug: Data Stats"):
    st.write(f"Total rows found for {market}: {len(df_market)}")
    st.write(df_market.tail(5))

# --- 4. PREPARE DATA ---
if len(df_market) > 2:
    df_market['month'] = df_market['Price Date'].dt.month
    df_market['dayofweek'] = df_market['Price Date'].dt.dayofweek
    df_market['price_lag_1'] = df_market['Modal_Price'].shift(1)
    df_market['price_lag_7'] = df_market['Modal_Price'].shift(7)
    df_final = df_market.dropna()

    # --- 5. WEATHER SIMULATION SLIDERS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Weather 'What-If'")
    
    # Defaults from the last known data point
    latest_row = df_market.iloc[[-1]]
    curr_temp = float(latest_row['Temperature_Avg (°C)'].fillna(25).values[0])
    curr_rain = float(latest_row['Rainfall (mm)'].fillna(0).values[0])
    curr_hum = float(latest_row['Humidity (%)'].fillna(50).values[0])

    sim_temp = st.sidebar.slider("Simulated Temp (°C)", 10.0, 50.0, curr_temp)
    sim_rain = st.sidebar.slider("Simulated Rainfall (mm)", 0.0, 500.0, curr_rain)
    sim_hum = st.sidebar.slider("Simulated Humidity (%)", 10.0, 100.0, curr_hum)

    if len(df_final) > 5:
        # Sensitivity Model (Weather Only)
        feats = ['month', 'dayofweek', 'Temperature_Avg (°C)', 'Rainfall (mm)', 'Humidity (%)']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df_final[feats], df_final['Modal_Price'])
        
        # Calculations
        base_pred = model.predict(latest_row[feats])[0]
        
        sim_row = latest_row[feats].copy()
        sim_row['Temperature_Avg (°C)'] = sim_temp
        sim_row['Rainfall (mm)'] = sim_rain
        sim_row['Humidity (%)'] = sim_hum
        sim_pred = model.predict(sim_row)[0]

        # --- 7. DISPLAY ---
        st.title(f"🌾 {commodity} in {market}")
        
        c1, c2, c3 = st.columns(3)
        actual_p = latest_row['Modal_Price'].values[0]
        c1.metric("Market Price", f"₹{actual_p:.2f}")
        c2.metric("Simulated Price", f"₹{sim_pred:.2f}")
        c3.metric("Weather Impact", f"₹{sim_pred - actual_p:.2f}", delta=f"{sim_pred - actual_p:.2f}")

        st.plotly_chart(px.line(df_market, x='Price Date', y='Modal_Price', title="Historical Price Trend"), use_container_width=True)
    else:
        st.warning("Found data, but not enough to train the AI (Minimum 5 days required after lags).")
else:
    st.error(f"No historical price data found for {commodity} in {market}. Try a different market.")
