import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriPrice Predictor", layout="wide")
st.title("🌾 Agricultural Price Prediction Dashboard")
st.markdown("Predicting market prices using Historical Data and Weather context.")

# --- LOAD DATA ---
@st.cache_data # This keeps the website fast by only loading data once
def load_data():
    df = pd.read_csv('Agri_Weather_Combined.csv')
    df['Price Date'] = pd.to_datetime(df['Price Date'])
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Select Parameters")
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
    
    # Train model on the fly for the selected market
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    X = df_final[features]
    y = df_final['Modal_Price']
    model.fit(X, y)
    
    # Get the latest data point to predict "Tomorrow"
    latest_data = df_final.iloc[[-1]]
    prediction = model.predict(latest_data[features])[0]
    
    # --- UI DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Next Price", f"₹{prediction:.2f}")
    col2.metric("Current Market Price", f"₹{latest_data['Modal_Price'].values[0]:.2f}")
    
    # Simple logic for price trend
    diff = prediction - latest_data['Modal_Price'].values[0]
    col3.metric("Expected Change", f"₹{diff:.2f}", delta=f"{diff:.2f}")

    # --- VISUALIZATION ---
    st.subheader(f"Price Trend in {market}, {state}")
    fig = px.line(df_market, x='Price Date', y='Modal_Price', 
                 title=f"Historical Price of {commodity}",
                 labels={'Modal_Price': 'Price (₹)', 'Price Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Weather impact section
    st.subheader("Weather Context")
    w_col1, w_col2 = st.columns(2)
    w_col1.write(f"**Average Temperature:** {latest_data['Temperature_Avg (°C)'].values[0]}°C")
    w_col2.write(f"**Recent Rainfall:** {latest_data['Rainfall (mm)'].values[0]} mm")

else:
    st.error("Not enough historical data for this specific market to make a prediction.")