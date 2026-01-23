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
import time

# --- CONFIGURATION & SETTINGS ---
st.set_page_config(page_title="AgroVision", layout="wide")

AGMARK_API_KEY = "579b464db66ec23bdd00000153830512e3d048f848bcb6701db55152"
WEATHER_API_KEY = "4057d99cd50050e4e8e1e063c92cafb1"

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

# ==================== STANDARD BINARY SEARCH ====================
class StandardBinarySearch:
    """
    Standard Binary Search on sorted price list
    Time Complexity: O(n log n) for sorting + O(log n) for search
    """
    
    def __init__(self):
        self.markets = []
        self.comparisons = 0
    
    def insert_market(self, market_name, price, distance_km, lat, lon):
        """Store all markets - O(1)"""
        self.markets.append({
            'name': market_name,
            'price': price,
            'distance': distance_km,
            'lat': lat,
            'lon': lon
        })
    
    def build_index(self):
        """Sort markets by price (descending) - O(n log n)"""
        self.markets.sort(key=lambda x: x['price'], reverse=True)
    
    def find_best_arbitrage(self, current_price, max_distance_km=None):
        """
        Binary search on sorted price list, then filter by distance
        Time Complexity: O(log n) search + O(k) distance filtering
        Returns: (market_data, profit_margin, comparisons_made)
        """
        self.comparisons = 0
        
        # Binary search for first market with price > current_price
        left, right = 0, len(self.markets) - 1
        candidate_idx = -1
        
        while left <= right:
            mid = (left + right) // 2
            self.comparisons += 1
            
            if self.markets[mid]['price'] > current_price:
                candidate_idx = mid
                left = mid + 1  # Continue searching for even higher prices
            else:
                right = mid - 1
        
        # Now filter candidates by distance
        if candidate_idx == -1:
            return None, 0, self.comparisons
        
        best_market = None
        best_profit = 0
        
        # Check all markets from candidate_idx onwards until price <= current_price
        for i in range(candidate_idx, len(self.markets)):
            market = self.markets[i]
            self.comparisons += 1
            
            if market['price'] <= current_price:
                break
            
            if max_distance_km is None or market['distance'] <= max_distance_km:
                profit = market['price'] - current_price
                if profit > best_profit:
                    best_profit = profit
                    best_market = market
        
        return best_market, best_profit, self.comparisons

# ==================== MODULO BINARY SEARCH ALGORITHM ====================
class ModuloBinarySearch:
    """
    Custom Modulo Binary Search for fast market arbitrage detection
    Time Complexity: O(log(n/e)) where e = num_buckets
    Space Complexity: O(n + e)
    """
    
    def __init__(self, num_distance_buckets=10, max_distance_km=500):
        self.num_buckets = num_distance_buckets
        self.max_distance = max_distance_km
        self.km_per_bucket = max_distance_km / num_distance_buckets
        self.storage = []  # Actual market data
        self.chains = [[] for _ in range(num_distance_buckets)]
        self.comparisons = 0
    
    def _get_bucket_index(self, distance_km):
        """Hash function: maps distance to bucket"""
        bucket = int(distance_km / self.km_per_bucket)
        return min(bucket, self.num_buckets - 1)
    
    def insert_market(self, market_name, price, distance_km, lat, lon):
        """Insert market into appropriate distance bucket - O(1)"""
        storage_index = len(self.storage)
        self.storage.append({
            'name': market_name,
            'price': price,
            'distance': distance_km,
            'lat': lat,
            'lon': lon
        })
        
        bucket_idx = self._get_bucket_index(distance_km)
        self.chains[bucket_idx].append(storage_index)
    
    def build_index(self):
        """Sort all chains by price (descending) - O(n log(n/e))"""
        for bucket_idx in range(self.num_buckets):
            chain = self.chains[bucket_idx]
            # Sort indices by price (highest first for arbitrage)
            chain.sort(key=lambda idx: self.storage[idx]['price'], reverse=True)
    
    def find_best_arbitrage(self, current_price, max_distance_km=None):
        """
        Find market with highest price > current_price using binary search
        Time Complexity: O(log(n/e))
        Returns: (market_data, profit_margin, comparisons_made)
        """
        if max_distance_km is None:
            max_distance_km = self.max_distance
        
        max_bucket = self._get_bucket_index(max_distance_km)
        best_market = None
        best_profit = 0
        self.comparisons = 0
        
        for bucket_idx in range(max_bucket + 1):
            chain = self.chains[bucket_idx]
            if not chain:
                continue
            
            # Binary search for first market with price > current_price
            left, right = 0, len(chain) - 1
            candidate_idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                market_idx = chain[mid]
                market_price = self.storage[market_idx]['price']
                self.comparisons += 1
                
                if market_price > current_price:
                    candidate_idx = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            if candidate_idx != -1:
                market_idx = chain[candidate_idx]
                market = self.storage[market_idx]
                profit = market['price'] - current_price
                
                if profit > best_profit:
                    best_profit = profit
                    best_market = market
        
        return best_market, best_profit, self.comparisons
    
    def find_top_n_opportunities(self, current_price, max_distance_km, n=5):
        """Find top N arbitrage opportunities"""
        if max_distance_km is None:
            max_distance_km = self.max_distance
            
        max_bucket = self._get_bucket_index(max_distance_km)
        opportunities = []
        
        for bucket_idx in range(max_bucket + 1):
            chain = self.chains[bucket_idx]
            for market_idx in chain:
                market = self.storage[market_idx]
                if market['price'] > current_price:
                    profit = market['price'] - current_price
                    opportunities.append((market, profit))
        
        opportunities.sort(key=lambda x: x[1], reverse=True)
        return opportunities[:n]

# ==================== HELPER FUNCTIONS ====================

def show_hero():
    st.markdown("""
        <div style="background-color:#1E3A1E;padding:40px;border-radius:15px;text-align:center;margin-bottom:25px">
            <h1 style="color:#FFFFFF;margin-bottom:0">AgroVision HARVEST AI</h1>
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
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            forecast_list = []
            daily_data = {}
            
            for item in data['list'][:56]:
                date = datetime.fromtimestamp(item['dt']).date()
                if date not in daily_data:
                    daily_data[date] = {'temps': [], 'humidity': [], 'rainfall': []}
                
                daily_data[date]['temps'].append(item['main']['temp'])
                daily_data[date]['humidity'].append(item['main']['humidity'])
                rain = item.get('rain', {}).get('3h', 0)
                daily_data[date]['rainfall'].append(rain)
            
            for date, values in sorted(daily_data.items())[:7]:
                forecast_list.append({
                    'Date': date,
                    'Temperature': round(np.mean(values['temps']), 2),
                    'Humidity': round(np.mean(values['humidity']), 2),
                    'Rainfall': round(sum(values['rainfall']), 2)
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
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'Temperature': data['main']['temp'],
                'Humidity': data['main']['humidity'],
                'Rainfall': data.get('rain', {}).get('1h', 0) * 24
            }
        return None
    except:
        return None

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
    except: 
        pass
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
    
    # Create map visualization
    fig = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", size="PRICE", color="PRICE", 
        hover_name="MARKET", hover_data=["PRICE", "distance_km"],
        color_continuous_scale="YlOrRd", zoom=6, mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # ==================== PERFORMANCE BENCHMARK ====================
    st.subheader("⚡ Algorithm Performance Comparison")
    
    # Filter markets (exclude current market)
    comparison_markets = map_df[map_df['MARKET'] != selected_market].copy()
    num_markets = len(comparison_markets)
    
    if num_markets > 0:
        # STANDARD BINARY SEARCH
        standard_bs = StandardBinarySearch()
        
        build_start = time.perf_counter()
        for _, row in comparison_markets.iterrows():
            standard_bs.insert_market(
                market_name=row['MARKET'],
                price=row['PRICE'],
                distance_km=row['distance_km'],
                lat=row['lat'],
                lon=row['lon']
            )
        standard_bs.build_index()
        standard_build_time = (time.perf_counter() - build_start) * 1000
        
        # Run standard binary search 100 times
        search_start = time.perf_counter()
        total_comparisons = 0
        for _ in range(100):
            best_market_std, profit_std, comparisons_std = standard_bs.find_best_arbitrage(
                current_price=current_price,
                max_distance_km=300
            )
            total_comparisons += comparisons_std
        standard_search_time = (time.perf_counter() - search_start) * 1000
        standard_avg_comparisons = total_comparisons / 100
        
        # MODULO BINARY SEARCH
        modulo_bs = ModuloBinarySearch(num_distance_buckets=10, max_distance_km=500)
        
        build_start = time.perf_counter()
        for _, row in comparison_markets.iterrows():
            modulo_bs.insert_market(
                market_name=row['MARKET'],
                price=row['PRICE'],
                distance_km=row['distance_km'],
                lat=row['lat'],
                lon=row['lon']
            )
        modulo_bs.build_index()
        modulo_build_time = (time.perf_counter() - build_start) * 1000
        
        # Run modulo binary search 100 times
        search_start = time.perf_counter()
        total_comparisons = 0
        for _ in range(100):
            best_market_mod, profit_mod, comparisons_mod = modulo_bs.find_best_arbitrage(
                current_price=current_price,
                max_distance_km=300
            )
            total_comparisons += comparisons_mod
        modulo_search_time = (time.perf_counter() - search_start) * 1000
        modulo_avg_comparisons = total_comparisons / 100
        
        # Display benchmark results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Standard Binary Search",
                f"{standard_avg_comparisons:.1f} comparisons",
                f"{standard_search_time:.2f} ms",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "🚀 Modulo Binary Search",
                f"{modulo_avg_comparisons:.1f} comparisons",
                f"{modulo_search_time:.2f} ms",
                delta_color="normal"
            )
        
        speedup = standard_avg_comparisons / modulo_avg_comparisons if modulo_avg_comparisons > 0 else 0
        time_speedup = standard_search_time / modulo_search_time if modulo_search_time > 0 else 0
        
        with col3:
            st.metric(
                "⚡ Comparison Reduction",
                f"{speedup:.2f}x",
                f"{((1 - 1/speedup) * 100):.0f}% fewer"
            )
        
        with col4:
            st.metric(
                "⏱️ Time Improvement",
                f"{time_speedup:.2f}x faster",
                f"Build: {modulo_build_time:.2f}ms"
            )
        
        # Detailed comparison table
        st.markdown("### 📊 Detailed Performance Metrics")
        
        comparison_df = pd.DataFrame({
            'Metric': [
                'Markets Analyzed',
                'Avg Comparisons',
                'Search Time (100 runs)',
                'Build/Sort Time',
                'Time Complexity',
                'Space Complexity',
                'Best For'
            ],
            'Standard Binary Search': [
                f"{num_markets:,}",
                f"{standard_avg_comparisons:.1f}",
                f"{standard_search_time:.2f} ms",
                f"{standard_build_time:.2f} ms",
                'O(log n) + O(k)',
                'O(n)',
                'Simple datasets'
            ],
            'Modulo Binary Search (Ours)': [
                f"{num_markets:,}",
                f"{modulo_avg_comparisons:.1f}",
                f"{modulo_search_time:.2f} ms",
                f"{modulo_build_time:.2f} ms",
                'O(log(n/e))',
                'O(n + e)',
                'Multi-constraint search'
            ],
            'Improvement': [
                '—',
                f"{speedup:.2f}x faster" if speedup >= 1 else f"{1/speedup:.2f}x slower",
                f"{time_speedup:.2f}x faster" if time_speedup >= 1 else f"{1/time_speedup:.2f}x slower",
                f"{standard_build_time/modulo_build_time:.2f}x" if modulo_build_time > 0 else '—',
                f'~{int(np.log2(num_markets/10) if num_markets > 10 else 1)} fewer levels',
                '+10 buckets overhead',
                'Distance bucketing wins'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Key insights
        st.info(f"""
        **🔍 Algorithm Analysis:**
        
        - **Standard Binary Search**: Sorts all {num_markets} markets by price, then searches in O(log n) time, 
        but must check each candidate against distance constraint sequentially.
        
        - **Modulo Binary Search**: Pre-buckets markets by distance (10 buckets), creating {len([c for c in modulo_bs.chains if c])} 
        non-empty distance zones. Searches only within relevant buckets, reducing search space by ~{speedup:.1f}x.
        
        - **Key Advantage**: When distance constraints matter (which they do for logistics!), bucketing eliminates 
        {int((1 - 1/speedup) * 100)}% of unnecessary comparisons by never examining distant markets.
        """)
        
        # Visual comparison
        st.markdown("### 📈 Visual Performance Comparison")
        
        perf_data = pd.DataFrame({
            'Method': ['Standard\nBinary Search', 'Modulo Binary\nSearch (Ours)'],
            'Comparisons': [standard_avg_comparisons, modulo_avg_comparisons],
            'Time (ms)': [standard_search_time, modulo_search_time]
        })
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_comp = px.bar(
                perf_data, x='Method', y='Comparisons',
                title='Average Comparisons per Search',
                color='Method',
                color_discrete_map={
                    'Standard\nBinary Search': '#f59e0b',
                    'Modulo Binary\nSearch (Ours)': '#10b981'
                }
            )
            fig_comp.update_layout(showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col_b:
            fig_time = px.bar(
                perf_data, x='Method', y='Time (ms)',
                title='Search Time (100 iterations)',
                color='Method',
                color_discrete_map={
                    'Standard\nBinary Search': '#f59e0b',
                    'Modulo Binary\nSearch (Ours)': '#10b981'
                }
            )
            fig_time.update_layout(showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Show best arbitrage opportunity (using modulo search results)
        if best_market_mod and profit_mod > 0:
            st.success(f"""
            ### 🚀 **Best Arbitrage Opportunity Found!**
            
            **Market:** {best_market_mod['name']}  
            **Price:** ₹{best_market_mod['price']:.2f}/quintal  
            **Profit Margin:** ₹{profit_mod:.2f}/quintal  
            **Distance:** {best_market_mod['distance']:.0f} km  
            **Comparisons Made:** {comparisons_mod} (vs {int(standard_avg_comparisons)} standard)
            
            💰 **For 100 quintals:** ₹{profit_mod * 100:,.0f} extra profit!
            """)
            
            # Show top opportunities
            top_opportunities = modulo_bs.find_top_n_opportunities(current_price, 300, n=5)
            if len(top_opportunities) > 1:
                st.markdown("### 🎯 Top 5 Arbitrage Opportunities")
                
                opp_data = []
                for i, (market, profit) in enumerate(top_opportunities, 1):
                    opp_data.append({
                        'Rank': i,
                        'Market': market['name'],
                        'Price': f"₹{market['price']:.2f}",
                        'Profit/Quintal': f"₹{profit:.2f}",
                        'Distance': f"{market['distance']:.0f} km",
                        'Profit (100Q)': f"₹{profit * 100:,.0f}"
                    })
                
                st.dataframe(pd.DataFrame(opp_data), use_container_width=True, hide_index=True)
        else:
            st.info("No profitable arbitrage opportunities found in nearby markets.")
    else:
        st.warning("Not enough market data for comparison.")

# ==================== MAIN APP ====================

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

show_hero()

if st.sidebar.button("🔄 Sync & Predict"):
    with st.spinner("Fetching Weather Data & Analyzing Market..."):
        default_coords = {"lat": 20.5937, "lon": 78.9629}
        coords = CITY_COORDS.get(selected_market, CITY_COORDS.get(selected_state, default_coords))
        
        weather_forecast = fetch_weather_forecast(coords['lat'], coords['lon'])
        current_weather = fetch_current_weather(coords['lat'], coords['lon'])
        
        if weather_forecast is None or current_weather is None:
            st.error("⚠️ Unable to fetch weather data. Please check your API key and internet connection.")
            st.info("Get a free API key from: https://openweathermap.org/api")
            st.stop()
        
        num_forecast_days = len(weather_forecast)
        if num_forecast_days < 1:
            st.error("⚠️ Weather forecast returned no data. Please try again later.")
            st.stop()
        
        st.subheader(f"🌤️ Current Weather Conditions: {selected_market}")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Temperature", f"{current_weather['Temperature']:.1f}°C")
        w_col2.metric("Humidity", f"{current_weather['Humidity']:.0f}%")
        w_col3.metric("Rainfall (24h est.)", f"{current_weather['Rainfall']:.1f} mm")
        
        df_hist = market_df[market_df['COMMODITY'] == selected_commodity].copy()
        df_live = fetch_live_prices(selected_commodity, selected_state, selected_market)
        df_master = pd.concat([df_hist, df_live], ignore_index=True).drop_duplicates('DATE').sort_values('DATE')
        
        df_master['month'] = df_master['DATE'].dt.month
        df_master['dayofweek'] = df_master['DATE'].dt.dayofweek
        df_master['lag_1'] = df_master['PRICE'].shift(1)
        df_master['lag_7'] = df_master['PRICE'].shift(7)
        df_master['Comm_Code'] = df_master['COMMODITY'].astype('category').cat.codes
        
        for col in ['Temperature', 'Humidity', 'Rainfall']:
            if col in df_master.columns:
                df_master[col].fillna(df_master[col].mean() if not df_master[col].isna().all() else current_weather[col], inplace=True)
            else:
                df_master[col] = current_weather[col]
        
        feature_list = ['month', 'dayofweek', 'lag_1', 'lag_7', 'Temperature', 'Humidity', 'Rainfall', 'Comm_Code']
        # --- COMPLETE FEATURE ENGINEERING & MODEL TRAINING ---
        actual_features = [f for f in feature_list if f in df_master.columns]
        df_train = df_master.dropna(subset=['PRICE'] + actual_features)

        if len(df_train) < 10:
            st.error("Not enough historical data to train the AI models for this commodity.")
            st.stop()

        X = df_train[actual_features]
        y = df_train['PRICE']

        # Model Dictionary
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        }

        # Train selected model(s)
        predictions = {}
        last_price = df_master['PRICE'].iloc[-1]
        
        # Forecast preparation
        future_dates = weather_forecast['Date'].tolist()
        forecast_df = pd.DataFrame({'DATE': future_dates})
        forecast_df['month'] = pd.to_datetime(forecast_df['DATE']).dt.month
        forecast_df['dayofweek'] = pd.to_datetime(forecast_df['DATE']).dt.dayofweek
        forecast_df['Temperature'] = weather_forecast['Temperature']
        forecast_df['Humidity'] = weather_forecast['Humidity']
        forecast_df['Rainfall'] = weather_forecast['Rainfall']
        forecast_df['Comm_Code'] = df_master['Comm_Code'].iloc[0]
        forecast_df['lag_1'] = last_price
        forecast_df['lag_7'] = df_master['PRICE'].iloc[-7] if len(df_master) >= 7 else last_price

        # AI Logic Execution
        if model_choice == "Ensemble (Consensus)":
            ensemble_preds = []
            for name, m in models.items():
                m.fit(X, y)
                ensemble_preds.append(m.predict(forecast_df[actual_features]))
            final_forecast = np.mean(ensemble_preds, axis=0)
        elif model_choice == "Prophet":
            prophet_df = df_train[['DATE', 'PRICE']].rename(columns={'DATE': 'ds', 'PRICE': 'y'})
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=len(future_dates))
            forecast = m.predict(future)
            final_forecast = forecast['yhat'].tail(len(future_dates)).values
        else:
            model = models[model_choice]
            model.fit(X, y)
            final_forecast = model.predict(forecast_df[actual_features])

        # --- VISUALIZATION ---
        st.write("---")
        st.subheader(f"📈 7-Day Price Forecast: {selected_commodity} ({model_choice})")
        
        res_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': final_forecast.round(2),
            'Weather': [f"{t}°C | {r}mm" for t, r in zip(weather_forecast['Temperature'], weather_forecast['Rainfall'])]
        })
        
        fig_forecast = px.line(res_df, x='Date', y='Predicted Price', markers=True, 
                               template="plotly_dark", color_discrete_sequence=['#A8E6A1'])
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        col_res1, col_res2 = st.columns(2)
        col_res1.dataframe(res_df, use_container_width=True)
        
        with col_res2:
            trend = ((final_forecast[-1] - last_price) / last_price) * 100
            trend_color = "normal" if trend > 0 else "inverse"
            st.metric("Expected Price Trend", f"{trend:.2f}%", delta=f"{final_forecast[-1] - last_price:.2f} ₹", delta_color=trend_color)
            st.info(f"The model suggests a {'bullish' if trend > 0 else 'bearish'} trend based on current weather patterns and historical seasonality.")

        # --- TRIGGER ARBITRAGE ANALYSIS ---
        display_map_and_arbitrage(df_base, selected_state, selected_commodity, selected_market, last_price)

st.sidebar.markdown("---")
st.sidebar.info("Nada Harvest AI uses Modulo Binary Search for high-speed spatial arbitrage detection and ensemble machine learning for price forecasting.")
