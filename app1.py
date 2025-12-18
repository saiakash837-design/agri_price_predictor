import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriPrice Predictor", layout="wide")
st.title("🌾 Agricultural Price Prediction Dashboard")
st.markdown("Predicting market prices using Historical Data and Weather context.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('Agri_Weather_Combined.csv')
    df['Price Date'] = pd.to_datetime(df['Price Date'])
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📍 Select Market")
state = st.sidebar.selectbox("Choose State", sorted(df['STATE'].unique()))
commodity = st.sidebar.selectbox("Choose Commodity", sorted(df[df['STATE'] == state]['Commodity'].unique()))

# Filter data based on selection
df_filtered = df[(df['STATE'] == state) & (df['Commodity'] == commodity)].copy()
market = st.sidebar.selectbox("Choose Market", sorted(df_filtered['Market Name'].unique()))

df_market = df_filtered[df_filtered['Market Name'] == market].sort_values('Price Date')

# --- DATA PREPARATION ---
df_market['month'] = df_market['Price Date'].dt.month
df_market['dayofweek'] = df_market['Price Date'].dt.dayofweek
df_market['price_lag_1'] = df_market['Modal_Price'].shift(1)
df_market['price_lag_7'] = df_market['Modal_Price'].shift(7)
df_final = df_market.dropna()

# --- TRAINING & PREDICTION ---
if len(df_final) > 10:
    features = ['month', 'dayofweek', 'price_lag_1', 'price_lag_7', 
                'Temperature_Avg (°C)', 'Rainfall (mm)', 'Humidity (%)']
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    X = df_final[features]
    y = df_final['Modal_Price']
    model.fit(X, y)
    
    # Get the latest data point
    latest_data = df_final.iloc[[-1]]
    prediction = model.predict(latest_data[features])[0]
    
    # --- UI DISPLAY: METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Next Price", f"₹{prediction:.2f}")
    col2.metric("Current Market Price", f"₹{latest_data['Modal_Price'].values[0]:.2f}")
    diff = prediction - latest_data['Modal_Price'].values[0]
    col3.metric("Expected Change", f"₹{diff:.2f}", delta=f"{diff:.2f}")

    # --- NEW: WHAT-IF SIMULATION (In Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Weather 'What-If' Simulation")
    st.sidebar.info("Adjust the sliders below to see how extreme weather would change today's price.")
    
    # Get current values to set as defaults for sliders
    curr_temp = float(latest_data['Temperature_Avg (°C)'].values[0])
    curr_rain = float(latest_data['Rainfall (mm)'].values[0])
    curr_hum = float(latest_data['Humidity (%)'].values[0])

    sim_temp = st.sidebar.slider("Simulated Temp (°C)", 10.0, 50.0, curr_temp)
    sim_rain = st.sidebar.slider("Simulated Rainfall (mm)", 0.0, 300.0, curr_rain)
    sim_hum = st.sidebar.slider("Simulated Humidity (%)", 10.0, 100.0, curr_hum)

    # Create simulation input
    sim_input = latest_data.copy()
    sim_input['Temperature_Avg (°C)'] = sim_temp
    sim_input['Rainfall (mm)'] = sim_rain
    sim_input['Humidity (%)'] = sim_hum
    
    sim_prediction = model.predict(sim_input[features])[0]

    # --- VISUALIZATION ---
    st.subheader(f"Price Trend in {market}, {state}")
    fig = px.line(df_market, x='Price Date', y='Modal_Price', 
                 title=f"Historical Price of {commodity}",
                 labels={'Modal_Price': 'Price (₹)', 'Price Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # --- SIMULATION RESULT BOX ---
    st.markdown("---")
    st.subheader("📊 Simulation Analysis")
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.write("### Simulated Prediction")
        st.title(f"₹{sim_prediction:.2f}")
        st.write(f"Based on: {sim_temp}°C, {sim_rain}mm Rain, {sim_hum}% Humidity")
    
    with sim_col2:
        impact = sim_prediction - prediction
        if impact > 0:
            st.warning(f"⚠️ This weather condition would likely **increase** the price by ₹{abs(impact):.2f}")
        elif impact < 0:
            st.success(f"✅ This weather condition would likely **decrease** the price by ₹{abs(impact):.2f}")
        else:
            st.write("No significant price change detected for these parameters.")

else:
    st.error("Not enough historical data for this specific market to make a prediction.")