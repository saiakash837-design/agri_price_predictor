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
from datetime import timedelta, datetime

# --- CONFIGURATION & SETTINGS ---
st.set_page_config(page_title="Nada Harvest AI", layout="wide", page_icon="🌾")

AGMARK_API_KEY = "579b464db66ec23bdd00000153830512e3d048f848bcb6701db55152"
# Add your OpenWeatherMap API key here (get free key from https://openweathermap.org/api)
WEATHER_API_KEY = "4057d99cd50050e4e8e1e063c92cafb1"  # Replace with your actual API key

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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Market Sync", "Active", "Government API")
    col2.metric("AI Models", "4 Engines", "Ensemble Active")
    col3.metric("Weather Integration", "Live", "7-Day Forecast")
    col4.metric("Coverage", "National", "All States")

@st.cache_data
def load_base_data():
    try:
        df = pd.read_csv('Agri_Weather_Lite.csv')
        df.columns = df.columns.str.strip()
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

@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat, lon):
    """
    Fetch 7-day weather forecast from OpenWeatherMap API
    Returns DataFrame with Temperature, Humidity, and Rainfall for next 7 days
    """
    try:
        # Using OneCall API 3.0 for 7-day forecast
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            forecast_list = []
            
            # Group by date and aggregate daily values
            daily_data = {}
            for item in data['list'][:56]:  # 7 days * 8 readings per day
                date = datetime.fromtimestamp(item['dt']).date()
                
                if date not in daily_data:
                    daily_data[date] = {
                        'temps': [],
                        'humidity': [],
                        'rainfall': []
                    }
                
                daily_data[date]['temps'].append(item['main']['temp'])
                daily_data[date]['humidity'].append(item['main']['humidity'])
                
                # Rainfall in mm (3h accumulated)
                rain = item.get('rain', {}).get('3h', 0)
                daily_data[date]['rainfall'].append(rain)
            
            # Calculate daily averages
            for date, values in sorted(daily_data.items())[:7]:
                forecast_list.append({
                    'Date': date,
                    'Temperature': round(np.mean(values['temps']), 2),
                    'Humidity': round(np.mean(values['humidity']), 2),
                    'Rainfall': round(sum(values['rainfall']), 2)  # Total daily rainfall
                })
            
            return pd.DataFrame(forecast_list)
        else:
            st.warning(f"Weather API Error: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Could not fetch weather data: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_current_weather(lat, lon):
    """
    Fetch current weather conditions
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'Temperature': data['main']['temp'],
                'Humidity': data['main']['humidity'],
                'Rainfall': data.get('rain', {}).get('1h', 0) * 24  # Convert to daily estimate
            }
        return None
    except:
        return None

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

from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, current_price):
    st.write("---")
    st.subheader(f"🌐 Regional Opportunity Map: {selected_commodity}")
    
    default_coords = {"lat": 20.5937, "lon": 78.9629}
    curr_coords = CITY_COORDS.get(selected_market, CITY_COORDS.get(selected_state, default_coords))
    
    map_df = df_base[(df_base['STATE'] == selected_state) & (df_base['COMMODITY'] == selected_commodity)].copy()
    if map_df.empty:
        st.warning("No regional data available for mapping.")
        return

    latest_date = map_df['DATE'].max()
    map_df = map_df[map_df['DATE'] == latest_date]

    map_df['lat'] = map_df['MARKET'].apply(lambda x: CITY_COORDS.get(x, CITY_COORDS.get(selected_state, default_coords))['lat'])
    map_df['lon'] = map_df['MARKET'].apply(lambda x: CITY_COORDS.get(x, CITY_COORDS.get(selected_state, default_coords))['lon'])
    
    map_df['lat'] += np.random.uniform(-0.05, 0.05, len(map_df))
    map_df['lon'] += np.random.uniform(-0.05, 0.05, len(map_df))

    map_df['distance_km'] = map_df.apply(lambda r: haversine(curr_coords['lat'], curr_coords['lon'], r['lat'], r['lon']), axis=1)
    
    fig = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", size="PRICE", color="PRICE", 
        hover_name="MARKET", hover_data=["PRICE", "distance_km"],
        color_continuous_scale="YlOrRd", zoom=6, mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    better_markets = map_df[(map_df['MARKET'] != selected_market) & (map_df['PRICE'] > current_price)].sort_values('PRICE', ascending=False)
    
    if not better_markets.empty:
        top_m = better_markets.iloc[0]
        st.success(f"🚀 **Arbitrage Opportunity:** **{top_m['MARKET']}** is offering **₹{top_m['PRICE'] - current_price:.2f}** more per quintal! (Approx. **{top_m['distance_km']:.0f} km** away)")

show_hero()

if st.sidebar.button("🔄 Sync & Predict"):
    with st.spinner("Fetching Weather Data & Analyzing Market..."):
        # Get coordinates for weather data
        default_coords = {"lat": 20.5937, "lon": 78.9629}
        coords = CITY_COORDS.get(selected_market, CITY_COORDS.get(selected_state, default_coords))
        
        # Fetch weather forecast
        weather_forecast = fetch_weather_forecast(coords['lat'], coords['lon'])
        current_weather = fetch_current_weather(coords['lat'], coords['lon'])
        
        if weather_forecast is None or current_weather is None:
            st.error("⚠️ Unable to fetch weather data. Please check your API key and internet connection.")
            st.info("Get a free API key from: https://openweathermap.org/api")
            st.stop()
        
        # Display current weather
        st.subheader(f"🌤️ Current Weather Conditions: {selected_market}")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Temperature", f"{current_weather['Temperature']:.1f}°C")
        w_col2.metric("Humidity", f"{current_weather['Humidity']:.0f}%")
        w_col3.metric("Rainfall (24h est.)", f"{current_weather['Rainfall']:.1f} mm")
        
        # Load historical and live price data
        df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
        df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
        df_master = pd.concat([df_hist, df_live], ignore_index=True).drop_duplicates('DATE').sort_values('DATE')
        
        # Feature Engineering with weather data
        df_master['month'] = df_master['DATE'].dt.month
        df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
        df_master['lag_1'] = df_master['PRICE'].shift(1)
        df_master['lag_7'] = df_master['PRICE'].shift(7)
        df_master['Comm_Code'] = df_master['COMMODITY'].astype('category').cat.codes
        
        # Fill missing weather data with historical averages or current conditions
        for col in ['Temperature', 'Humidity', 'Rainfall']:
            if col in df_master.columns:
                df_master[col].fillna(df_master[col].mean() if not df_master[col].isna().all() else current_weather[col], inplace=True)
            else:
                df_master[col] = current_weather[col]
        
        # Define features (weather included)
        feature_list = ['month', 'dayofweek', 'lag_1', 'lag_7', 'Temperature', 'Humidity', 'Rainfall', 'Comm_Code']
        actual_features = [f for f in feature_list if f in df_master.columns]
        
        df_ml = df_master.dropna(subset=['PRICE', 'lag_1', 'lag_7'])

        if len(df_ml) > 10:
            X, y = df_ml[actual_features], df_ml['PRICE']
            
            # Train Models
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X, y)
            lgb = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42).fit(X, y)
            
            # Prophet
            df_p = df_ml[['DATE', 'PRICE']].rename(columns={'DATE': 'ds', 'PRICE': 'y'})
            prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(df_p)

            # 7-Day Forecast with Weather Integration
            last_row = df_ml.iloc[-1]
            forecast_data = []
            c_rf = c_xgb = c_lgb = last_row['PRICE']
            p_preds = prophet_model.predict(prophet_model.make_future_dataframe(periods=7)).tail(7)['yhat'].values

            for i in range(7):
                f_date = last_row['DATE'] + timedelta(days=i+1)
                
                # Get weather forecast for this day
                weather_day = weather_forecast.iloc[i]
                
                # Build input features
                in_dict = {
                    'month': f_date.month,
                    'dayofweek': f_date.dayofweek,
                    'Temperature': weather_day['Temperature'],
                    'Humidity': weather_day['Humidity'],
                    'Rainfall': weather_day['Rainfall'],
                    'Comm_Code': last_row['Comm_Code']
                }
                
                # Add lag_7 if available
                if 'lag_7' in actual_features:
                    in_dict['lag_7'] = df_ml.iloc[-7]['PRICE'] if len(df_ml) > 7 else last_row['PRICE']
                
                # Predict with each model
                temp_df = pd.DataFrame([in_dict])
                
                temp_df['lag_1'] = c_rf
                p1 = rf.predict(temp_df[actual_features])[0]
                
                temp_df['lag_1'] = c_xgb
                p2 = xgb.predict(temp_df[actual_features])[0]
                
                temp_df['lag_1'] = c_lgb
                p3 = lgb.predict(temp_df[actual_features])[0]
                
                p4 = p_preds[i]
                
                ens = (p1 + p2 + p3 + p4) / 4
                mapping = {"Random Forest": p1, "XGBoost": p2, "LightGBM": p3, "Prophet": p4, "Ensemble (Consensus)": ens}
                
                forecast_data.append({
                    "Date": f_date.strftime('%Y-%m-%d'),
                    "Price": round(mapping[model_choice], 2),
                    "Temp (°C)": weather_day['Temperature'],
                    "Humidity (%)": weather_day['Humidity'],
                    "Rain (mm)": weather_day['Rainfall']
                })
                
                c_rf, c_xgb, c_lgb = p1, p2, p3

            # --- DISPLAY RESULTS ---
            st.title(f"📊 {selected_commodity} Insights")
            
            # Accuracy Plot
            df_ml['RF'], df_ml['XGB'], df_ml['LGB'] = rf.predict(X), xgb.predict(X), lgb.predict(X)
            df_ml['Prophet'] = prophet_model.predict(df_p)['yhat'].values
            df_ml['Ensemble'] = (df_ml['RF'] + df_ml['XGB'] + df_ml['LGB'] + df_ml['Prophet']) / 4
            
            col_map = {"Random Forest": "RF", "XGBoost": "XGB", "LightGBM": "LGB", "Prophet": "Prophet", "Ensemble (Consensus)": "Ensemble"}
            
            fig = px.line(df_ml.tail(30), x='DATE', y=['PRICE', col_map[model_choice]], 
                         title=f"Last 30 Days: Prediction vs Actual ({model_choice})",
                         labels={'value': 'Price (₹/Quintal)', 'DATE': 'Date'})
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)

            # Forecast Table and Metrics
            col_table, col_metrics = st.columns([2, 1])
            
            with col_table:
                st.subheader("📅 7-Day Weather-Based Forecast")
                st.dataframe(pd.DataFrame(forecast_data), use_container_width=True, height=300)

            with col_metrics:
                st.subheader("🎯 Model Performance")
                r2 = r2_score(y, df_ml[col_map[model_choice]])
                st.metric("R² Score", f"{r2:.3f}", f"{r2*100:.1f}% accuracy")
                
                avg_forecast = np.mean([f['Price'] for f in forecast_data])
                price_change = avg_forecast - last_row['PRICE']
                st.metric("7-Day Avg Forecast", f"₹{avg_forecast:.2f}", f"{price_change:+.2f}")
                
                st.metric("Data Points", len(df_ml))

            # Weather Impact Visualization
            st.subheader("🌦️ Weather Forecast Impact")
            weather_fig = px.line(pd.DataFrame(forecast_data), x='Date', 
                                 y=['Temp (°C)', 'Humidity (%)', 'Rain (mm)'],
                                 title="7-Day Weather Trends")
            st.plotly_chart(weather_fig, use_container_width=True)

            display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, last_row['PRICE'])
        else:
            st.warning("Not enough data to generate a forecast for this selection.")
else:
    st.info("👈 Select parameters and click **Sync & Predict** to see weather-integrated forecasts.")
