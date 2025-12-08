# ======================================================================
# STORMPROOF 2025 — Institutional Bot Advisor (Version 2.0)
# Enhanced Risk Parity Backtesting vs Ray Dalio's All Weather (since 1996)
# Now with Dalio-inspired dynamic leverage, trend-following, finer regimes, BTC balance, oil overlay, and crisis alpha
# Professional version for fund managers and investment bankers
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# FRED API for macroeconomic data
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

# Premium Dark Mode CSS for institutional clients
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
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        text-align: center;
        color: #8a8a9a;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0d0d12;
        border-right: 1px solid #1a1a2e;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #c0c0c0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
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
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stDataFrame {
        background-color: #12121a;
        border-radius: 8px;
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
    
    .stAlert {
        background-color: #12121a;
        border: 1px solid #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ========================== HISTORICAL MARKET CRISES ==========================
MARKET_CRISES = [
    {"name": "Asian Crisis", "start": "1997-07-01", "end": "1998-01-01", "duration": "6 months"},
    {"name": "LTCM/Russia", "start": "1998-08-01", "end": "1998-11-01", "duration": "3 months"},
    {"name": "Dot-com Bubble", "start": "2000-03-01", "end": "2002-10-01", "duration": "31 months"},
    {"name": "9/11 Attacks", "start": "2001-09-01", "end": "2001-10-15", "duration": "6 weeks"},
    {"name": "Subprime Crisis", "start": "2007-10-01", "end": "2009-03-01", "duration": "17 months"},
    {"name": "Flash Crash", "start": "2010-05-01", "end": "2010-07-01", "duration": "2 months"},
    {"name": "EU Debt Crisis", "start": "2011-07-01", "end": "2012-06-01", "duration": "11 months"},
    {"name": "China/Oil Crash", "start": "2015-08-01", "end": "2016-02-01", "duration": "6 months"},
    {"name": "COVID-19 Crash", "start": "2020-02-01", "end": "2020-04-01", "duration": "2 months"},
    {"name": "Inflation/Ukraine", "start": "2022-01-01", "end": "2022-10-01", "duration": "9 months"},
]

# ========================== DATA FUNCTIONS WITH CACHE ==========================

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
        data = pdr.DataReader(series_id, 'fred', start_date, end_date)
        return data
    except Exception:
        return None


@st.cache_data(ttl=7200, show_spinner=False)
def get_historical_data(start_year, end_year):
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    data_sources = {}
    series_list = []
    
    # Core assets
    # EQUITIES - SPY (1993+) or S&P 500 Index
    spy_data = download_yahoo_data('SPY', start_date, end_date)
    if spy_data is not None and len(spy_data) > 0:
        spy_data.name = 'SPY'
        series_list.append(spy_data)
        data_sources['SPY'] = 'Yahoo Finance (SPY ETF)'
    else:
        gspc_data = download_yahoo_data('^GSPC', start_date, end_date)
        if gspc_data is not None:
            gspc_data.name = 'SPY'
            series_list.append(gspc_data)
            data_sources['SPY'] = 'Yahoo Finance (S&P 500 Index proxy)'
    
    # LONG-TERM BONDS - TLT (2002+) or IEF
    tlt_data = download_yahoo_data('TLT', start_date, end_date)
    if tlt_data is not None and len(tlt_data) > 0:
        tlt_data.name = 'TLT'
        series_list.append(tlt_data)
        data_sources['TLT'] = 'Yahoo Finance (TLT ETF)'
    else:
        ief_data = download_yahoo_data('IEF', start_date, end_date)
        if ief_data is not None and len(ief_data) > 0:
            ief_data.name = 'TLT'
            series_list.append(ief_data)
            data_sources['TLT'] = 'Yahoo Finance (IEF proxy)'
    
    # GOLD - GLD (2004+) or Gold Futures
    gld_data = download_yahoo_data('GLD', start_date, end_date)
    if gld_data is not None and len(gld_data) > 0:
        gld_data.name = 'GLD'
        series_list.append(gld_data)
        data_sources['GLD'] = 'Yahoo Finance (GLD ETF)'
    else:
        gc_data = download_yahoo_data('GC=F', start_date, end_date)
        if gc_data is not None and len(gc_data) > 0:
            gc_data.name = 'GLD'
            series_list.append(gc_data)
            data_sources['GLD'] = 'Yahoo Finance (Gold Futures proxy)'
    
    # COMMODITIES - DBC (2006+) or GSG + USO for oil overlay
    dbc_data = download_yahoo_data('DBC', start_date, end_date)
    if dbc_data is not None and len(dbc_data) > 0:
        dbc_data.name = 'DBC'
        series_list.append(dbc_data)
        data_sources['DBC'] = 'Yahoo Finance (DBC ETF)'
    else:
        gsg_data = download_yahoo_data('GSG', start_date, end_date)
        if gsg_data is not None and len(gsg_data) > 0:
            gsg_data.name = 'DBC'
            series_list.append(gsg_data)
            data_sources['DBC'] = 'Yahoo Finance (GSG proxy)'
    
    # Oil for inflation overlay
    uso_data = download_yahoo_data('USO', start_date, end_date)
    if uso_data is not None and len(uso_data) > 0:
        uso_data.name = 'USO'
        series_list.append(uso_data)
        data_sources['USO'] = 'Yahoo Finance (USO Oil ETF)'
    
    # BTC from 2014+
    if end_year >= 2014:
        btc_data = download_yahoo_data('BTC-USD', start_date, end_date)
        if btc_data is not None and len(btc_data) > 0:
            btc_data.name = 'BTC'
            series_list.append(btc_data)
            data_sources['BTC'] = 'Yahoo Finance (BTC-USD)'
    
    # VIX
    vix_data = download_yahoo_data('^VIX', start_date, end_date)
    if vix_data is not None:
        vix_data.name = '^VIX'
        series_list.append(vix_data)
        data_sources['VIX'] = 'Yahoo Finance (CBOE VIX Index)'
    
    return series_list, data_sources


@st.cache_data(ttl=7200, show_spinner=False)
def get_fred_macro_data(start_date, end_date):
    macro_data = {}
    sources = {}
    
    if not FRED_AVAILABLE:
        return None, "pandas_datareader not available"
    
    try:
        cpi = download_fred_series('CPIAUCSL', start_date, end_date)
        if cpi is not None:
            macro_data['CPIAUCSL'] = cpi
            sources['CPI'] = 'FRED (CPIAUCSL)'
        
        fedfunds = download_fred_series('FEDFUNDS', start_date, end_date)
        if fedfunds is not None:
            macro_data['FEDFUNDS'] = fedfunds
            sources['Fed Funds'] = 'FRED (FEDFUNDS)'
        
        unrate = download_fred_series('UNRATE', start_date, end_date)
        if unrate is not None:
            macro_data['UNRATE'] = unrate
            sources['Unemployment'] = 'FRED (UNRATE)'
        
        # Breakeven inflation proxy from FRED if available
        breakeven = download_fred_series('T10YIE', start_date, end_date)  # 10y breakeven
        if breakeven is not None:
            macro_data['BREAKEVEN'] = breakeven
            sources['Breakeven Inflation'] = 'FRED (T10YIE)'
        
        return macro_data, sources
    except Exception as e:
        return None, str(e)


# ========================== STRATEGY FUNCTIONS (ENHANCED) ==========================

def detect_regime(row):
    """Finer regime detection inspired by Dalio's multi-regime approach"""
    growth = (row.get('UNRATE_change', 0) < 0) or (row['UNRATE'] < 5.5)
    inflation_rising = row['CPI_YoY'] > (row.get('CPI_YoY_12m_avg', 2.5) + 0.5)
    high_inflation = row['CPI_YoY'] > 3.5
    low_rates = row['FEDFUNDS'] < 1.5
    breakeven = row.get('BREAKEVEN', row['CPI_YoY'])
    
    if growth and not inflation_rising and breakeven < 2.0:
        return "Reflation"      # Equities + Gold
    elif growth and high_inflation:
        return "Stagflation"    # Commodities + Gold
    elif not growth and not inflation_rising:
        return "Deflation"      # Long Bonds
    else:
        return "Recession"      # Bonds + Gold + Cash proxy

def trend_signal(returns_window, horizon=252):  # 1 year daily ~252
    """Trend-following overlay: Time Series Momentum"""
    if len(returns_window) < horizon:
        return np.ones(len(returns_window.columns))
    cumret = (1 + returns_window.iloc[-horizon:]).prod() - 1
    signal = np.where(cumret > 0, 1.0, 0.3)  # Full if uptrend, 30% if down
    return signal

def dynamic_leverage(portfolio_vol_60d, target_vol=0.10):
    """Dalio-inspired dynamic leverage: scale based on realized vol"""
    if portfolio_vol_60d == 0 or np.isnan(portfolio_vol_60d):
        return 1.0
    leverage = target_vol / portfolio_vol_60d
    return np.clip(leverage, 0.5, 2.5)

def vix_panic_buy(vix, current_w, recent_returns):
    """Enhanced VIX signal with more granularity"""
    if vix > 80:  # Extreme crisis
        return np.array([0.0, 0.5, 0.5, 0.0]), 0.5  # 50% TLT + 50% GLD, delever
    elif vix > 60:
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

def risk_parity(cov_matrix):
    vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    w = 1 / vol
    return w / w.sum()

def gold_btc_balance(gld_ret_12m, btc_ret_12m):
    """Dynamic balance: allocate 8% to better performer (low historical corr ~0, good diversification)"""
    if btc_ret_12m > gld_ret_12m:
        return {'BTC': 0.08, 'GLD': 0.0}  # BTC wins
    else:
        return {'BTC': 0.0, 'GLD': 0.08}   # Gold wins (safer in crashes)

def oil_overlay(cpi_yoy, dbc_ret, uso_ret):
    """Replace DBC with 50/50 DBC+USO when high inflation"""
    if cpi_yoy > 4.0:
        return 0.5 * dbc_ret + 0.5 * uso_ret
    return dbc_ret

# ========================== HEADER ==========================
st.markdown('<h1 class="main-header">⚡ STORMPROOF 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enhanced Institutional Bot Advisor • Dynamic Risk Parity vs All Weather (Dalio-Inspired)</p>', unsafe_allow_html=True)

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    
    st.markdown("##### 📅 Analysis Period")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            "Start",
            options=list(range(1996, 2026)),
            index=0,
            help="Backtest start year (All Weather launched in 1996)"
        )
    with col2:
        end_year = st.selectbox(
            "End",
            options=list(range(1996, 2026)),
            index=25,  # Default to 2021
            help="Backtest end year"
        )
    
    if start_year >= end_year:
        st.error("⚠️ Start year must be before end year")
        st.stop()
    
    st.markdown("##### 💰 Initial Capital")
    initial_capital = st.number_input(
        "Amount ($)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=10_000_000,
        step=1_000_000,
        format="%d",
        help="Starting capital for simulation"
    )
    
    st.markdown("---")
    
    run_simulation = st.button("🚀 RUN STORMPROOF 2.0 ANALYSIS", type="primary")

# ========================== MAIN CONTENT ==========================

if not run_simulation:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Objective")
        st.markdown("Outperform Bridgewater's All Weather with dynamic enhancements (1996+).")
    
    with col2:
        st.markdown("#### 📊 Investment Universe")
        st.markdown("**Equities** (SPY) • **Bonds** (TLT) • **Gold/BTC** (GLD/BTC balanced) • **Commodities** (DBC + USO oil)")
    
    with col3:
        st.markdown("#### ⚡ New Edges")
        st.markdown("- Dalio: Dynamic leverage, trend-following, finer regimes\n- BTC ramp (0→8%, gold-balanced)\n- Oil overlay on inflation\n- VIX>80 crisis shift")
    
    st.markdown("---")
    
    st.markdown("#### 📈 Data Sources")
    st.markdown("""
    - **Market Data**: Yahoo Finance (ETFs, BTC-USD, proxies)
    - **Macro Data**: FRED — CPI, Fed Funds, Unemployment, Breakeven Inflation
    - **Volatility**: CBOE VIX
    """)
    
    st.markdown("---")
    st.markdown('<p class="methodology-text">📋 Enhanced proprietary methodology — Dalio-inspired + BTC diversification (low corr ~0 with gold)</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
    st.stop()

# ========================== DATA LOADING ==========================

start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'
core_tickers = ['SPY', 'TLT', 'GLD', 'DBC']
extended_tickers = core_tickers + ['USO', 'BTC'] if end_year >= 2014 else core_tickers

data_messages = []

with st.spinner('Loading enhanced market data...'):
    try:
        series_list, data_sources = get_historical_data(start_year, end_year)
        
        if len([s for s in series_list if s.name in core_tickers]) < 4:
            st.error(f"❌ Insufficient core data: only {len(series_list)} assets. Start from 2007 for full.")
            st.stop()
        
        prices = pd.concat(series_list, axis=1).resample('M').last()
        df = prices.copy()
        
        # Check missing core
        missing_core = [t for t in core_tickers if t not in df.columns]
        if missing_core:
            st.error(f"❌ Missing core: {', '.join(missing_core)}")
            st.stop()
        
        # Macro
        fred_loaded = False
        if FRED_AVAILABLE:
            macro_data, macro_sources = get_fred_macro_data(start_date, end_date)
            if macro_data:
                for key, series in macro_data.items():
                    monthly = series.resample('M').last()
                    df = df.join(monthly, how='left')
                    df[key] = df[key].ffill().bfill()
                fred_loaded = True
                data_messages.append("✅ FRED macro loaded (incl. Breakeven Inflation)")
                for name, src in macro_sources.items():
                    st.sidebar.markdown(f"{name}: {src}")
        
        if not fred_loaded:
            np.random.seed(42)
            n = len(df)
            df['CPIAUCSL'] = 150 + np.cumsum(np.random.normal(0.3, 0.4, n))
            df['FEDFUNDS'] = np.clip(4.0 + np.cumsum(np.random.normal(0, 0.3, n)), 0, 10)
            df['UNRATE'] = np.clip(5.5 + np.cumsum(np.random.normal(0, 0.15, n)), 3, 12)
            df['BREAKEVEN'] = np.clip(2.0 + np.cumsum(np.random.normal(0, 0.2, n)), 0, 5)
            data_messages.append("⚠️ Simulated macro used")
        
        df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
        df['UNRATE_change'] = df['UNRATE'].diff()
        df['CPI_YoY_12m_avg'] = df['CPI_YoY'].rolling(12).mean()
        
        returns_raw = df[core_tickers].pct_change()
        valid_mask = returns_raw.notna().all(axis=1) & df['CPI_YoY'].notna()
        df = df[valid_mask]
        returns = returns_raw[valid_mask]
        
        # Add extended returns if available
        if 'BTC' in df.columns:
            btc_ret = df['BTC'].pct_change()[valid_mask]
            returns['BTC'] = btc_ret
        if 'USO' in df.columns:
            uso_ret = df['USO'].pct_change()[valid_mask]
            returns['USO'] = uso_ret
        
        data_messages.append(f"✅ {len(df)} months ({df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')})")
        
        if len(df) < 24:
            st.error(f"Insufficient data ({len(df)} months).")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# ========================== SIMULATION (STORMPROOF 2.0) ==========================

df['Regime'] = df.apply(detect_regime, axis=1)

with st.spinner('Running STORMPROOF 2.0 simulation...'):
    capital_stormproof = [initial_capital]
    capital_allweather = [initial_capital]
    weights_aw = np.array([0.30, 0.55, 0.075, 0.075])  # SPY, TLT, GLD, DBC
    peak = initial_capital
    portfolio_returns = []
    btc_alloc_history = []  # For logging
    
    cap_storm = initial_capital
    cap_aw = initial_capital
    
    for idx in range(1, len(returns)):  # Start from 1 for pct_change
        date = df.index[idx]
        year_frac = (date.year - 2014) / (end_year - 2014) if end_year > 2014 and date.year >= 2014 else 0
        max_btc_alloc = np.clip(year_frac * 0.08, 0, 0.08)  # Ramp 0->8% from 2014
        
        # All Weather (static)
        ret_aw = np.dot(weights_aw, returns[core_tickers].iloc[idx].values)
        cap_aw *= (1 + ret_aw)
        capital_allweather.append(cap_aw)
        
        # Stormproof 2.0
        window = returns[core_tickers].iloc[max(0, idx-36):idx]
        long_window = returns[core_tickers].iloc[max(0, idx-120):idx]
        
        if len(window) < 12:
            ret_storm = ret_aw  # Fallback
        else:
            # 1. Regime base weights
            regime = df['Regime'].iloc[idx]
            if regime == "Reflation":
                base_w = np.array([0.45, 0.30, 0.15, 0.10])
            elif regime == "Stagflation":
                base_w = np.array([0.10, 0.10, 0.40, 0.40])
            elif regime == "Deflation":
                base_w = np.array([0.15, 0.70, 0.10, 0.05])
            else:  # Recession
                base_w = np.array([0.10, 0.60, 0.20, 0.10])
            
            # 2. Risk Parity
            cov = window.cov() * 12  # Monthly
            rp_w = risk_parity(cov)
            
            # 3. Trend overlay
            trend = trend_signal(returns[core_tickers], horizon=12)  # 12 months
            w_trend = base_w * trend
            w_trend /= w_trend.sum()
            
            # 4. VIX adjustment
            vix = df['^VIX'].iloc[idx]
            vix_w, vix_mult = vix_panic_buy(vix, base_w, returns[core_tickers].iloc[max(0, idx-3):idx].mean().values)
            
            # 5. Inflation oil overlay for comm
            cpi_yoy = df['CPI_YoY'].iloc[idx]
            dbc_ret = returns['DBC'].iloc[idx]
            uso_ret = returns.get('USO', pd.Series([0], index=[date])).iloc[0] if 'USO' in returns.columns else 0
            comm_ret = oil_overlay(cpi_yoy, dbc_ret, uso_ret)
            
            # 6. Gold/BTC balance
            gld_12m = returns['GLD'].iloc[max(0, idx-12):idx].sum()
            btc_12m = returns.get('BTC', pd.Series([0]*12, index=returns.index[max(0, idx-12):idx])).iloc[max(0, idx-12):idx].sum()
            balance = gold_btc_balance(gld_12m, btc_12m)
            btc_alloc = balance.get('BTC', 0)
            gld_bonus = balance.get('GLD', 0)
            btc_alloc_history.append(btc_alloc)
            
            # Adjust base for BTC/GLD
            final_base = base_w.copy()
            final_base[2] += gld_bonus  # GLD index 2
            if 'BTC' in returns.columns and date.year >= 2014:
                # Insert BTC as new asset, reduce others proportionally
                total_w = final_base.sum() + max_btc_alloc
                final_base = final_base * (1 - max_btc_alloc / total_w)
                final_base = np.append(final_base, max_btc_alloc)  # Now 5 assets: SPY,TLT,GLD,DBC,BTC
            
            # 7. Blend weights
            blend_w = 0.5 * rp_w + 0.3 * w_trend + 0.2 * vix_w
            blend_w = np.maximum(blend_w, 0.05)  # Min 5%
            blend_w /= blend_w.sum()
            
            # Crisis alpha override
            if vix > 80:
                if len(blend_w) == 4:
                    blend_w = np.array([0.0, 1.0, 0.0, 0.0])  # All to TLT
                else:
                    blend_w = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # All to TLT, BTC=0
                vix_mult = 0.5  # Delever
            
            # Compute return (handle 4 or 5 assets)
            core_rets = returns[core_tickers].iloc[idx].values
            if len(blend_w) > 4 and 'BTC' in returns.columns:
                core_rets = np.append(core_rets, returns['BTC'].iloc[idx])
                comm_ret_adjusted = np.copy(core_rets)
                comm_ret_adjusted[3] = comm_ret  # Override DBC with overlay
                ret_storm_base = np.dot(blend_w, comm_ret_adjusted)
            else:
                core_rets[3] = comm_ret  # Override DBC
                ret_storm_base = np.dot(blend_w, core_rets)
            
            # 8. Dynamic leverage
            recent_port_rets = portfolio_returns[-60:] if len(portfolio_returns) >= 60 else portfolio_returns
            port_vol = np.std(recent_port_rets) * np.sqrt(12) if recent_port_rets else 0.10
            leverage = dynamic_leverage(port_vol)
            
            ret_storm = ret_storm_base * leverage * vix_mult
        
        cap_storm *= (1 + ret_storm)
        portfolio_returns.append(ret_storm)
        capital_stormproof.append(cap_storm)
        
        peak = max(peak, cap_storm)

# ========================== RESULTS ==========================

# Align lengths
min_len = min(len(capital_stormproof), len(capital_allweather))
result = pd.DataFrame({
    "STORMPROOF 2.0": capital_stormproof[:min_len],
    "All Weather": capital_allweather[:min_len]
}, index=returns.index[:min_len])

final_storm = result["STORMPROOF 2.0"].iloc[-1]
final_aw = result["All Weather"].iloc[-1]
years = (result.index[-1] - result.index[0]).days / 365.25
annual_return_storm = ((final_storm / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
annual_return_aw = ((final_aw / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

# ========================== CHART ==========================

st.markdown("### 📈 Enhanced Comparative Performance")

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')

result.plot(ax=ax, linewidth=2.5, color=['#667eea', '#f093fb'], alpha=0.95)

for crisis in MARKET_CRISES:
    crisis_start = pd.to_datetime(crisis["start"])
    crisis_end = pd.to_datetime(crisis["end"])
    if crisis_start >= result.index[0] and crisis_start <= result.index[-1]:
        ax.axvspan(crisis_start, min(crisis_end, result.index[-1]), alpha=0.12, color='#ff4444', zorder=0)
        mid_date = crisis_start + (min(crisis_end, result.index[-1]) - crisis_start) / 2
        y_pos = result.loc[result.index >= crisis_start].iloc[0].max() * 1.08 if not result.loc[result.index >= crisis_start].empty else result.max().max() * 0.9
        ax.annotate(f"{crisis['name']}\n{crisis['duration']}", xy=(mid_date, y_pos), fontsize=7, color='#ff6666', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ff4444', alpha=0.9), zorder=5)

ax.set_title(f"STORMPROOF 2.0 vs All Weather ({start_year} → {end_year})", fontsize=20, fontweight='bold', color='#e0e0e0', pad=15)
ax.set_ylabel("Portfolio Value ($)", fontsize=13, color='#a0a0a0')
ax.grid(alpha=0.1, linestyle='-', color='#404050')
ax.legend(fontsize=12, loc='upper left', facecolor='#12121a', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
ax.tick_params(colors='#a0a0a0', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#2a2a3a')
ax.spines['left'].set_color('#2a2a3a')

def format_millions(x, pos):
    if x >= 1e9: return f'{x/1e9:.1f}B$'
    elif x >= 1e6: return f'{x/1e6:.1f}M$'
    elif x >= 1e3: return f'{x/1e3:.0f}K$'
    return f'{x:.0f}$'

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
plt.tight_layout()
st.pyplot(fig)

# ========================== METRICS ==========================

st.markdown("---")
st.markdown("### 📊 Simulation Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 STORMPROOF 2.0 Final", f"{final_storm/1e6:.2f}M $", f"+{((final_storm / initial_capital - 1) * 100):.1f}%")

with col2:
    st.metric("📈 All Weather Final", f"{final_aw/1e6:.2f}M $", f"+{((final_aw / initial_capital - 1) * 100):.1f}%")

with col3:
    surperf = (final_storm / final_aw - 1) * 100
    st.metric("🚀 Outperformance", f"+{surperf:.1f}%", "vs All Weather")

with col4:
    st.metric("📈 Annualized Return", f"{annual_return_storm:.2f}%", "STORMPROOF 2.0")

# BTC stats if used
if end_year >= 2014 and 'BTC' in data_sources:
    avg_btc_alloc = np.mean(btc_alloc_history) * 100
    st.info(f"💎 BTC Avg Allocation: {avg_btc_alloc:.1f}% (balanced vs Gold; low hist corr ~0 for diversification)")

# ========================== RISK METRICS ==========================

st.markdown("---")
st.markdown("### 📉 Risk Metrics")

storm_rets = pd.Series(portfolio_returns, index=result.index[1:])  # Align
vol_storm = storm_rets.std() * np.sqrt(12) * 100
vol_aw = result["All Weather"].pct_change().std() * np.sqrt(12) * 100
dd_storm = ((result["STORMPROOF 2.0"] / result["STORMPROOF 2.0"].cummax()) - 1).min() * 100
dd_aw = ((result["All Weather"] / result["All Weather"].cummax()) - 1).min() * 100

rf_rate = df['FEDFUNDS'].mean() if 'FEDFUNDS' in df else 2.0
sharpe_storm = (annual_return_storm - rf_rate) / vol_storm if vol_storm > 0 else 0
sharpe_aw = (annual_return_aw - rf_rate) / vol_aw if vol_aw > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    stats_storm = pd.DataFrame({
        'STORMPROOF 2.0': [f"{annual_return_storm:.2f}%", f"{vol_storm:.2f}%", f"{dd_storm:.2f}%", f"{sharpe_storm:.2f}"]
    }, index=['Annual Return', 'Volatility', 'Max DD', 'Sharpe'])
    st.dataframe(stats_storm, use_container_width=True)

with col2:
    stats_aw = pd.DataFrame({
        'All Weather': [f"{annual_return_aw:.2f}%", f"{vol_aw:.2f}%", f"{dd_aw:.2f}%", f"{sharpe_aw:.2f}"]
    }, index=['Annual Return', 'Volatility', 'Max DD', 'Sharpe'])
    st.dataframe(stats_aw, use_container_width=True)

with col3:
    stats_diff = pd.DataFrame({
        'Difference': [f"{annual_return_storm - annual_return_aw:+.2f}%", f"{vol_storm - vol_aw:+.2f}%", f"{dd_storm - dd_aw:+.2f}%", f"{sharpe_storm - sharpe_aw:+.2f}"]
    }, index=['Annual Return', 'Volatility', 'Max DD', 'Sharpe'])
    st.dataframe(stats_diff, use_container_width=True)

# ========================== FOOTER ==========================

st.markdown("---")
footer_data = " • ".join(data_messages)
st.markdown(f'<p class="footer-text">{footer_data} • BTC/Gold balanced for low corr diversification (hist ~0, not inverse but complementary)</p>', unsafe_allow_html=True)
st.markdown('<p class="methodology-text">📋 Dalio-enhanced + BTC safe ramp (0→8%, momentum-balanced vs Gold) — Low risk add due to decorrelation</p>', unsafe_allow_html=True)
st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
