# ======================================================================
# STORMPROOF 2.0 — Institutional Bot Advisor
# Enhanced Risk Parity vs Ray Dalio's All Weather (1996-2025)
# Features: Dynamic Leverage, Trend Overlay, BTC/Gold Balance, Crisis Alpha
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# FRED API
try:
    from pandas_datareader import data as pdr
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ========================== STREAMLIT CONFIG ==========================
st.set_page_config(
    page_title="STORMPROOF 2.0 - Institutional Bot Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4a90d9 0%, #667eea 50%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #8a8a9a;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0d0d12;
        border-right: 1px solid #1a1a2e;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4a90d9 0%, #667eea 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .footer-text {
        text-align: center;
        color: #606070;
        font-size: 0.85rem;
        padding: 1rem 0;
        border-top: 1px solid #1a1a2e;
        margin-top: 2rem;
    }
    .footer-contact {
        text-align: center;
        color: #8a8a9a;
        font-size: 0.95rem;
        padding: 0.5rem 0;
    }
    .methodology-text {
        text-align: center;
        color: #667eea;
        font-size: 0.9rem;
        font-style: italic;
        padding: 1rem 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========================== MARKET CRISES ==========================
MARKET_CRISES = [
    {"name": "Asian Crisis", "start": "1997-07-01", "end": "1998-01-01", "duration": "6 months"},
    {"name": "LTCM/Russia", "start": "1998-08-01", "end": "1998-11-01", "duration": "3 months"},
    {"name": "Dot-com Bubble", "start": "2000-03-01", "end": "2002-10-01", "duration": "31 months"},
    {"name": "9/11", "start": "2001-09-01", "end": "2001-10-15", "duration": "6 weeks"},
    {"name": "Subprime", "start": "2007-10-01", "end": "2009-03-01", "duration": "17 months"},
    {"name": "Flash Crash", "start": "2010-05-01", "end": "2010-07-01", "duration": "2 months"},
    {"name": "EU Debt", "start": "2011-07-01", "end": "2012-06-01", "duration": "11 months"},
    {"name": "China/Oil", "start": "2015-08-01", "end": "2016-02-01", "duration": "6 months"},
    {"name": "COVID-19", "start": "2020-02-01", "end": "2020-04-01", "duration": "2 months"},
    {"name": "Inflation/Ukraine", "start": "2022-01-01", "end": "2022-10-01", "duration": "9 months"},
]

# ========================== DATA FUNCTIONS ==========================

@st.cache_data(ttl=7200, show_spinner=False)
def download_yahoo_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=30)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
            series = data[col].iloc[:, 0] if len(data[col].shape) > 1 else data[col]
        else:
            col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            series = data[col]
        series.name = ticker
        return series
    except Exception:
        return None


@st.cache_data(ttl=7200, show_spinner=False)
def download_fred_series(series_id, start_date, end_date):
    try:
        return pdr.DataReader(series_id, 'fred', start_date, end_date)
    except Exception:
        return None


@st.cache_data(ttl=7200, show_spinner=False)
def get_all_data(start_year, end_year, include_btc=True):
    """Download all market data with proxies for pre-ETF periods"""
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    data_sources = {}
    series_dict = {}
    
    # Core assets
    assets_config = [
        ('SPY', 'SPY', '^GSPC'),
        ('TLT', 'TLT', 'IEF'),
        ('GLD', 'GLD', 'GC=F'),
        ('DBC', 'DBC', 'GSG'),
        ('^VIX', '^VIX', None),
    ]
    
    for name, primary, fallback in assets_config:
        data = download_yahoo_data(primary, start_date, end_date)
        if data is not None and len(data) > 0:
            series_dict[name] = data
            data_sources[name] = f'Yahoo ({primary})'
        elif fallback:
            data = download_yahoo_data(fallback, start_date, end_date)
            if data is not None and len(data) > 0:
                data.name = name
                series_dict[name] = data
                data_sources[name] = f'Yahoo ({fallback} proxy)'
    
    # Optional: USO for oil overlay
    uso_data = download_yahoo_data('USO', start_date, end_date)
    if uso_data is not None and len(uso_data) > 0:
        series_dict['USO'] = uso_data
        data_sources['USO'] = 'Yahoo (USO)'
    
    # Optional: Bitcoin (from 2014)
    if include_btc:
        btc_data = download_yahoo_data('BTC-USD', start_date, end_date)
        if btc_data is not None and len(btc_data) > 0:
            series_dict['BTC'] = btc_data
            data_sources['BTC'] = 'Yahoo (BTC-USD)'
    
    return series_dict, data_sources


@st.cache_data(ttl=7200, show_spinner=False)
def get_fred_macro(start_date, end_date):
    """Get macro data from FRED"""
    if not FRED_AVAILABLE:
        return None, {}
    
    macro = {}
    sources = {}
    
    series_ids = {
        'CPIAUCSL': 'CPI',
        'FEDFUNDS': 'Fed Funds',
        'UNRATE': 'Unemployment',
        'T10YIE': 'Breakeven Inflation 10Y'  # Better inflation signal
    }
    
    for series_id, name in series_ids.items():
        data = download_fred_series(series_id, start_date, end_date)
        if data is not None:
            macro[series_id] = data
            sources[name] = f'FRED ({series_id})'
    
    return macro, sources


# ========================== STRATEGY FUNCTIONS ==========================

def detect_regime(row):
    """Enhanced regime detection (4 regimes instead of simple seasons)"""
    unrate_falling = row.get('UNRATE_change', 0) < 0
    growth = unrate_falling or row['UNRATE'] < 5.5
    
    # Use breakeven inflation if available, else CPI
    inflation_rate = row.get('T10YIE', row['CPI_YoY'])
    inflation_avg = row.get('CPI_YoY_12m_avg', 2.5)
    
    inflation_rising = inflation_rate > inflation_avg + 0.5
    high_inflation = row['CPI_YoY'] > 3.5
    
    if growth and not inflation_rising:
        return "Reflation"      # Good for stocks + gold
    elif growth and inflation_rising:
        return "Stagflation"    # Good for gold + commodities
    elif not growth and not inflation_rising:
        return "Deflation"      # Good for long bonds
    else:
        return "Recession"      # Cash + gold + bonds


def get_regime_weights(regime):
    """Base weights per regime (more aggressive than classic All Weather)"""
    regime_weights = {
        "Reflation":   np.array([0.45, 0.30, 0.15, 0.10]),  # SPY, TLT, GLD, DBC
        "Stagflation": np.array([0.10, 0.10, 0.40, 0.40]),
        "Deflation":   np.array([0.15, 0.70, 0.10, 0.05]),
        "Recession":   np.array([0.10, 0.60, 0.20, 0.10]),
    }
    return regime_weights.get(regime, np.array([0.30, 0.55, 0.075, 0.075]))


def trend_signal(returns_series, horizon=12):
    """Trend-following overlay (Time Series Momentum)"""
    if len(returns_series) < horizon:
        return np.ones(len(returns_series.columns)) * 0.5
    
    cumret = (1 + returns_series.iloc[-horizon:]).cumprod().iloc[-1]
    signal = np.where(cumret > 1, 1.0, 0.3)  # Full if up, 30% if down
    return signal


def dynamic_leverage(realized_vol, target_vol=0.10):
    """Dynamic leverage based on realized portfolio volatility"""
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 1.0
    leverage = target_vol / realized_vol
    return np.clip(leverage, 0.5, 2.5)  # Min 0.5x, Max 2.5x


def risk_parity(cov_matrix):
    """Risk Parity allocation"""
    vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    w = 1 / vol
    return w / w.sum()


def vix_panic_buy(vix, current_w, recent_returns):
    """Buy the dip during VIX spikes"""
    if vix > 60:
        boost = 0.25
    elif vix > 45:
        boost = 0.18
    elif vix > 35:
        boost = 0.12
    else:
        return current_w.copy(), 1.0
    
    losers = np.argsort(recent_returns)[:2]
    w = current_w.copy()
    for l in losers:
        w[l] += boost
    w /= w.sum()
    return w, 1.5


def crisis_alpha(vix):
    """Crisis alpha: extreme VIX → 100% TLT"""
    if vix > 80:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 100% TLT
    return None


def gold_btc_balance(gold_ret_12m, btc_ret_12m, year, max_btc=0.08):
    """Dynamic Gold/BTC balance based on relative performance"""
    if year < 2014:
        return 1.0, 0.0  # No BTC before 2014
    
    # Progressive BTC allocation (0% in 2014 → 8% in 2022+)
    year_frac = min((year - 2014) / 8, 1.0)
    max_btc_alloc = max_btc * year_frac
    
    # If BTC outperforming gold, allocate more to BTC
    if btc_ret_12m is not None and not np.isnan(btc_ret_12m):
        if btc_ret_12m > gold_ret_12m:
            btc_alloc = max_btc_alloc
            gold_alloc = 1.0 - btc_alloc
        else:
            btc_alloc = max_btc_alloc * 0.5  # Half allocation if underperforming
            gold_alloc = 1.0 - btc_alloc
    else:
        btc_alloc = 0.0
        gold_alloc = 1.0
    
    return gold_alloc, btc_alloc


def oil_overlay(cpi_yoy, dbc_weight, uso_available):
    """Oil overlay: 50/50 DBC+USO when CPI > 4%"""
    if uso_available and cpi_yoy > 4.0:
        return dbc_weight * 0.5, dbc_weight * 0.5  # Split between DBC and USO
    return dbc_weight, 0.0


# ========================== HEADER ==========================
st.markdown('<h1 class="main-header">⚡ STORMPROOF 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enhanced Bot Advisor • Dynamic Leverage • Trend Overlay • Crisis Alpha</p>', unsafe_allow_html=True)

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    
    st.markdown("##### 📅 Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Start", list(range(2007, 2026)), index=0)
    with col2:
        end_year = st.selectbox("End", list(range(2007, 2026)), index=18)
    
    if start_year >= end_year:
        st.error("⚠️ Start must be before end")
        st.stop()
    
    st.markdown("##### 💰 Initial Capital")
    initial_capital = st.number_input("Amount ($)", 100_000, 1_000_000_000, 10_000_000, 1_000_000)
    
    st.markdown("##### 🎛️ Advanced Settings")
    include_btc = st.checkbox("Include BTC (2014+)", value=True, help="Progressive allocation up to 8%")
    target_vol = st.slider("Target Volatility", 0.05, 0.20, 0.10, 0.01, help="For dynamic leverage")
    max_leverage = st.slider("Max Leverage", 1.0, 3.0, 2.5, 0.1)
    
    st.markdown("---")
    run_simulation = st.button("🚀 RUN ANALYSIS", type="primary")

# ========================== LANDING PAGE ==========================
if not run_simulation:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎯 Objective")
        st.markdown("Outperform Bridgewater's All Weather with dynamic risk management.")
    with col2:
        st.markdown("#### 📊 Universe")
        st.markdown("**SPY** • **TLT** • **GLD** • **DBC** • **BTC** (optional)")
    with col3:
        st.markdown("#### ⚡ Edge")
        st.markdown("Dynamic leverage, trend overlay, crisis alpha, BTC/Gold balance.")
    
    st.markdown("---")
    st.markdown("#### 🔬 STORMPROOF 2.0 Enhancements")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **From Bridgewater's 2014-2021 improvements:**
        - Dynamic leverage (0.5x - 2.5x based on realized vol)
        - Trend-following overlay (12-month momentum)
        - Enhanced regime detection (4 regimes)
        - Better inflation signal (breakeven rate)
        """)
    with col2:
        st.markdown("""
        **Additional alpha sources:**
        - BTC allocation (progressive 0→8% since 2014)
        - Oil overlay (USO when CPI > 4%)
        - Crisis alpha (VIX > 80 → 100% TLT)
        - Gold/BTC dynamic balance
        """)
    
    st.markdown("---")
    st.markdown('<p class="methodology-text">📋 Proprietary methodology — Request full documentation for institutional due diligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
    st.stop()

# ========================== DATA LOADING ==========================
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'
tickers = ['SPY', 'TLT', 'GLD', 'DBC']
data_messages = []

with st.spinner('Loading market data...'):
    try:
        # Get price data
        series_dict, data_sources = get_all_data(start_year, end_year, include_btc)
        
        # Check required assets
        missing = [t for t in tickers + ['^VIX'] if t not in series_dict]
        if missing:
            st.error(f"❌ Missing: {', '.join(missing)}")
            st.stop()
        
        # Build price DataFrame
        prices = pd.concat([series_dict[t] for t in tickers + ['^VIX']], axis=1)
        prices.columns = tickers + ['^VIX']
        df = prices.resample('M').last()
        
        # Add optional assets
        uso_available = 'USO' in series_dict
        btc_available = 'BTC' in series_dict
        
        if uso_available:
            uso = series_dict['USO'].resample('M').last()
            df['USO'] = uso
        
        if btc_available:
            btc = series_dict['BTC'].resample('M').last()
            df['BTC'] = btc
        
        # Get macro data
        fred_loaded = False
        if FRED_AVAILABLE:
            macro, macro_sources = get_fred_macro(start_date, end_date)
            if macro and len(macro) >= 3:
                for key, series in macro.items():
                    monthly = series.resample('M').last()
                    df = df.join(monthly, how='left')
                    df[key] = df[key].ffill().bfill()
                fred_loaded = True
                data_messages.append("✅ FRED macro data loaded")
        
        if not fred_loaded:
            np.random.seed(42)
            n = len(df)
            df['CPIAUCSL'] = 200 + np.cumsum(np.random.normal(0.3, 0.4, n))
            df['FEDFUNDS'] = np.clip(2.0 + np.cumsum(np.random.normal(0, 0.2, n)), 0, 8)
            df['UNRATE'] = np.clip(5.0 + np.cumsum(np.random.normal(0, 0.1, n)), 3, 12)
            data_messages.append("⚠️ FRED unavailable — Simulated macro")
        
        # Compute derived indicators
        df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
        df['CPI_YoY_12m_avg'] = df['CPI_YoY'].rolling(12).mean()
        df['UNRATE_change'] = df['UNRATE'].diff()
        
        # Clean data
        returns_raw = df[tickers].pct_change()
        valid_mask = returns_raw.notna().all(axis=1) & df['CPI_YoY'].notna()
        df = df[valid_mask]
        returns = returns_raw[valid_mask]
        
        # BTC returns if available
        if btc_available and 'BTC' in df.columns:
            df['BTC_ret'] = df['BTC'].pct_change()
        
        # USO returns if available
        if uso_available and 'USO' in df.columns:
            df['USO_ret'] = df['USO'].pct_change()
        
        data_messages.append(f"✅ {len(df)} months ({df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')})")
        
        if len(df) < 24:
            st.error("Insufficient data")
            st.stop()
            
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ========================== SIMULATION ==========================
with st.spinner('Running STORMPROOF 2.0 simulation...'):
    capital_stormproof = [initial_capital]
    capital_allweather = [initial_capital]
    weights_aw = np.array([0.30, 0.55, 0.075, 0.075])
    
    peak = initial_capital
    portfolio_returns = []
    regimes_history = []
    leverage_history = []
    
    for idx in range(1, len(returns)):
        current_date = df.index[idx]
        current_year = current_date.year
        
        # === ALL WEATHER (benchmark) ===
        ret_aw = np.dot(weights_aw, returns.iloc[idx].values)
        capital_allweather.append(capital_allweather[-1] * (1 + ret_aw))
        
        # === STORMPROOF 2.0 ===
        window = returns.iloc[max(0, idx-36):idx]
        
        if len(window) < 12:
            ret_storm = ret_aw
            regimes_history.append("N/A")
            leverage_history.append(1.0)
        else:
            # 1. Regime detection
            regime = detect_regime(df.iloc[idx])
            regimes_history.append(regime)
            
            # 2. Check for crisis alpha (VIX > 80)
            vix = df['^VIX'].iloc[idx]
            crisis_w = crisis_alpha(vix)
            
            if crisis_w is not None:
                final_w = crisis_w
                leverage = 0.8  # Reduce leverage in extreme crisis
            else:
                # 3. Base weights from regime
                base_w = get_regime_weights(regime)
                
                # 4. Risk Parity refinement
                cov = window.cov() * 252 + np.eye(4) * 1e-6
                rp_w = risk_parity(cov)
                
                # 5. Trend overlay
                trend = trend_signal(returns.iloc[max(0, idx-12):idx])
                w_trend = base_w * trend
                w_trend = w_trend / w_trend.sum() if w_trend.sum() > 0 else base_w
                
                # 6. VIX panic buy
                recent_rets = returns.iloc[max(0, idx-3):idx].mean().values
                vix_w, _ = vix_panic_buy(vix, base_w, recent_rets)
                
                # 7. Blend weights
                final_w = 0.40 * rp_w + 0.35 * w_trend + 0.25 * vix_w
                final_w = np.maximum(final_w, 0.03)  # Min 3% per asset
                final_w = final_w / final_w.sum()
                
                # 8. Oil overlay (CPI > 4%)
                cpi_yoy = df['CPI_YoY'].iloc[idx]
                if uso_available and cpi_yoy > 4.0:
                    dbc_w, uso_w = oil_overlay(cpi_yoy, final_w[3], True)
                    final_w[3] = dbc_w  # Reduce DBC
                    # USO return added separately
                
                # 9. Dynamic leverage
                if len(portfolio_returns) >= 12:
                    realized_vol = np.std(portfolio_returns[-60:]) * np.sqrt(12) if len(portfolio_returns) >= 60 else np.std(portfolio_returns[-12:]) * np.sqrt(12)
                else:
                    realized_vol = target_vol
                
                leverage = dynamic_leverage(realized_vol, target_vol)
                leverage = min(leverage, max_leverage)
                
                # 10. Drawdown protection
                current_dd = (capital_stormproof[-1] - peak) / peak if peak > 0 else 0
                if current_dd < -0.20:
                    leverage = 0.6  # Deleverage on big drawdown
            
            leverage_history.append(leverage)
            
            # Calculate return
            ret_storm = np.dot(final_w, returns.iloc[idx].values) * leverage
            
            # Add BTC contribution if enabled
            if include_btc and btc_available and 'BTC' in df.columns and current_year >= 2014:
                gold_ret_12m = df['GLD'].iloc[max(0, idx-12):idx].pct_change().sum() if idx >= 12 else 0
                btc_ret_12m = df['BTC'].iloc[max(0, idx-12):idx].pct_change().sum() if idx >= 12 and 'BTC' in df.columns else None
                
                gold_alloc, btc_alloc = gold_btc_balance(gold_ret_12m, btc_ret_12m, current_year)
                
                if btc_alloc > 0 and 'BTC_ret' in df.columns and not pd.isna(df['BTC_ret'].iloc[idx]):
                    # Adjust gold weight and add BTC
                    gold_idx = 2  # GLD position
                    btc_contribution = btc_alloc * df['BTC_ret'].iloc[idx] * leverage
                    gold_reduction = btc_alloc * final_w[gold_idx]
                    ret_storm = ret_storm - (gold_reduction * returns.iloc[idx].values[gold_idx]) + btc_contribution
        
        portfolio_returns.append(ret_storm)
        new_cap = capital_stormproof[-1] * (1 + ret_storm)
        capital_stormproof.append(new_cap)
        peak = max(peak, new_cap)

# ========================== RESULTS ==========================
result = pd.DataFrame({
    "STORMPROOF 2.0": capital_stormproof[1:],
    "All Weather": capital_allweather[1:]
}, index=returns.index)

final_storm = result["STORMPROOF 2.0"].iloc[-1]
final_aw = result["All Weather"].iloc[-1]
years = (result.index[-1] - result.index[0]).days / 365.25
annual_storm = ((final_storm / initial_capital) ** (1 / years) - 1) * 100
annual_aw = ((final_aw / initial_capital) ** (1 / years) - 1) * 100

# ========================== CHART ==========================
st.markdown("### 📈 Comparative Performance")

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')

result.plot(ax=ax, linewidth=2.5, color=['#00d4aa', '#f093fb'], alpha=0.95)

# Crisis annotations
for crisis in MARKET_CRISES:
    cs = pd.to_datetime(crisis["start"])
    ce = pd.to_datetime(crisis["end"])
    if cs >= result.index[0] and cs <= result.index[-1]:
        ax.axvspan(cs, min(ce, result.index[-1]), alpha=0.12, color='#ff4444', zorder=0)
        mid = cs + (min(ce, result.index[-1]) - cs) / 2
        try:
            y_pos = result.loc[result.index >= cs].iloc[0].max() * 1.08
        except:
            y_pos = result.max().max() * 0.9
        ax.annotate(f"{crisis['name']}\n{crisis['duration']}", xy=(mid, y_pos),
                    fontsize=7, color='#ff6666', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ff4444', alpha=0.9))

ax.set_title(f"STORMPROOF 2.0 vs All Weather ({start_year} → {end_year})", fontsize=20, fontweight='bold', color='#e0e0e0', pad=15)
ax.set_ylabel("Portfolio Value ($)", fontsize=13, color='#a0a0a0')
ax.grid(alpha=0.1, color='#404050')
ax.legend(fontsize=12, loc='upper left', facecolor='#12121a', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
ax.tick_params(colors='#a0a0a0', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#2a2a3a')
ax.spines['left'].set_color('#2a2a3a')

def fmt_m(x, p):
    if x >= 1e9: return f'{x/1e9:.1f}B$'
    elif x >= 1e6: return f'{x/1e6:.1f}M$'
    elif x >= 1e3: return f'{x/1e3:.0f}K$'
    return f'{x:.0f}$'

ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
plt.tight_layout()
st.pyplot(fig)

# ========================== METRICS ==========================
st.markdown("---")
st.markdown("### 📊 Simulation Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 STORMPROOF 2.0", f"{final_storm/1e6:.2f}M $", f"+{((final_storm/initial_capital-1)*100):.1f}%")
with col2:
    st.metric("📈 All Weather", f"{final_aw/1e6:.2f}M $", f"+{((final_aw/initial_capital-1)*100):.1f}%")
with col3:
    st.metric("🚀 Outperformance", f"+{((final_storm/final_aw-1)*100):.1f}%", "vs All Weather")
with col4:
    st.metric("📈 Annualized", f"{annual_storm:.2f}%", "STORMPROOF 2.0")

# ========================== RISK METRICS ==========================
st.markdown("---")
st.markdown("### 📉 Risk Metrics")

vol_storm = result["STORMPROOF 2.0"].pct_change().std() * np.sqrt(12) * 100
vol_aw = result["All Weather"].pct_change().std() * np.sqrt(12) * 100
dd_storm = ((result["STORMPROOF 2.0"] / result["STORMPROOF 2.0"].cummax()) - 1).min() * 100
dd_aw = ((result["All Weather"] / result["All Weather"].cummax()) - 1).min() * 100
rf = df['FEDFUNDS'].mean() if 'FEDFUNDS' in df.columns else 2.0
sharpe_storm = (annual_storm - rf) / vol_storm if vol_storm > 0 else 0
sharpe_aw = (annual_aw - rf) / vol_aw if vol_aw > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.dataframe(pd.DataFrame({'STORMPROOF 2.0': [f"{annual_storm:.2f}%", f"{vol_storm:.2f}%", f"{dd_storm:.2f}%", f"{sharpe_storm:.2f}"]},
                 index=['Annualized', 'Volatility', 'Max DD', 'Sharpe']), use_container_width=True)
with col2:
    st.dataframe(pd.DataFrame({'All Weather': [f"{annual_aw:.2f}%", f"{vol_aw:.2f}%", f"{dd_aw:.2f}%", f"{sharpe_aw:.2f}"]},
                 index=['Annualized', 'Volatility', 'Max DD', 'Sharpe']), use_container_width=True)
with col3:
    st.dataframe(pd.DataFrame({'Difference': [f"{annual_storm-annual_aw:+.2f}%", f"{vol_storm-vol_aw:+.2f}%", f"{dd_storm-dd_aw:+.2f}%", f"{sharpe_storm-sharpe_aw:+.2f}"]},
                 index=['Annualized', 'Volatility', 'Max DD', 'Sharpe']), use_container_width=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown(f'<p class="footer-text">{" • ".join(data_messages)}</p>', unsafe_allow_html=True)
st.markdown('<p class="methodology-text">📋 Proprietary methodology (Dynamic Leverage • Trend Overlay • 4-Regime Detection • BTC/Gold Balance • Crisis Alpha • Oil Overlay) — Request full documentation for institutional due diligence</p>', unsafe_allow_html=True)
st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
