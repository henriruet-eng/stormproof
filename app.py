# ======================================================================
# STORMPROOF — Institutional Bot Advisor
# Full Historical Backtest vs All Weather (1996-2025)
# LOCAL DATA VERSION - No API timeouts, instant loading
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

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

# ========================== LOCAL DATA LOADING ==========================

@st.cache_data(ttl=86400)
def load_local_data():
    """
    Load pre-computed historical data from local CSV files.
    Data covers 1996-2025 with realistic market scenarios.
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load market data
        market_path = os.path.join(script_dir, 'market_data.csv')
        market_df = pd.read_csv(market_path, index_col='Date', parse_dates=True)
        
        # Load macro data
        macro_path = os.path.join(script_dir, 'macro_data.csv')
        macro_df = pd.read_csv(macro_path, index_col='Date', parse_dates=True)
        
        # Load BTC data
        btc_path = os.path.join(script_dir, 'btc_data.csv')
        btc_df = pd.read_csv(btc_path, index_col='Date', parse_dates=True)
        
        return market_df, macro_df, btc_df, True
        
    except Exception as e:
        st.error(f"Error loading local data: {e}")
        return None, None, None, False


# ========================== STRATEGY FUNCTIONS ==========================

def detect_regime(row):
    """Enhanced 4-regime detection"""
    unrate_change = row.get('UNRATE_change', 0)
    growth = unrate_change < 0 or row['UNRATE'] < 5.5
    
    inflation_rate = row.get('CPI_YoY', 2.5)
    inflation_avg = row.get('CPI_YoY_12m_avg', 2.5)
    inflation_rising = inflation_rate > inflation_avg + 0.5
    
    if growth and not inflation_rising:
        return "Reflation"
    elif growth and inflation_rising:
        return "Stagflation"
    elif not growth and not inflation_rising:
        return "Deflation"
    return "Recession"


def get_regime_weights(regime):
    """Base weights per regime"""
    weights = {
        "Reflation":   np.array([0.45, 0.30, 0.15, 0.10]),
        "Stagflation": np.array([0.10, 0.10, 0.40, 0.40]),
        "Deflation":   np.array([0.15, 0.70, 0.10, 0.05]),
        "Recession":   np.array([0.10, 0.60, 0.20, 0.10]),
    }
    return weights.get(regime, np.array([0.30, 0.55, 0.075, 0.075]))


def elastic_tension(row):
    """Market tension indicator"""
    tension = 0
    if row['VIX'] > 40:
        tension += 0.4
    if abs(row.get('CPI_YoY', 2)) > 4:
        tension += 0.3
    if row.get('FEDFUNDS', 2) < 1:
        tension += 0.2
    return min(tension, 1.0)


def trend_signal(returns_series, horizon=12):
    """Trend-following overlay"""
    if len(returns_series) < horizon:
        return np.ones(4) * 0.5
    cumret = (1 + returns_series.iloc[-horizon:]).cumprod().iloc[-1]
    return np.where(cumret > 1, 1.0, 0.3)


def dynamic_leverage(realized_vol, target_vol=0.10, max_lev=2.5):
    """Dynamic leverage based on realized volatility"""
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 1.0
    return np.clip(target_vol / realized_vol, 0.5, max_lev)


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
        return np.array([0.0, 1.0, 0.0, 0.0])
    return None


def causal_adjustment(regime, cpi_change, fed_change):
    """Pearl causal inference adjustment"""
    adj = np.zeros(4)
    if "Stagflation" in regime or "Recession" in regime:
        adj[2] += 0.15
    if "Recession" in regime and fed_change < -0.5:
        adj[1] += 0.25
    if cpi_change > 1:
        adj[3] += 0.10
    return adj


def quantum_monte_carlo(returns_window, n_sim=1000):
    """Quantum-inspired Monte Carlo simulation"""
    mu = returns_window.mean().values
    cov = returns_window.cov().values * 252
    cov += np.eye(len(cov)) * 1e-6
    
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < 0):
        cov = cov + np.eye(len(cov)) * (abs(eigvals.min()) + 1e-6)
    
    try:
        sim = np.random.multivariate_normal(mu, cov, n_sim)
    except np.linalg.LinAlgError:
        std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
        sim = np.random.normal(mu, std, (n_sim, len(mu)))
    
    sums = np.clip(np.sum(sim, axis=1), -50, 50)
    amplitudes = np.sqrt(np.abs(np.exp(sums)))
    amp_sum = amplitudes.sum()
    amplitudes = amplitudes / amp_sum if amp_sum > 0 else np.ones(n_sim) / n_sim
    
    best_idx = np.argsort(-amplitudes)[:100]
    return sim[best_idx].mean(axis=0)


def risk_parity(cov_matrix):
    """Risk Parity allocation"""
    vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    w = 1 / vol
    return w / w.sum()


def gold_btc_balance(gold_ret_12m, btc_ret_12m, year, max_btc=0.08):
    """Dynamic Gold/BTC balance"""
    if year < 2014:
        return 1.0, 0.0
    
    year_frac = min((year - 2014) / 8, 1.0)
    max_btc_alloc = max_btc * year_frac
    
    if btc_ret_12m is not None and not np.isnan(btc_ret_12m):
        btc_alloc = max_btc_alloc if btc_ret_12m > gold_ret_12m else max_btc_alloc * 0.5
    else:
        btc_alloc = 0.0
    
    return 1.0 - btc_alloc, btc_alloc


# ========================== HEADER ==========================
st.markdown('<h1 class="main-header">⚡ STORMPROOF 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Institutional Bot Advisor • Full Historical Backtest vs All Weather (1996-2025)</p>', unsafe_allow_html=True)

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
            help="1996 = All Weather launch year"
        )
    with col2:
        end_year = st.selectbox(
            "End",
            options=list(range(1996, 2026)),
            index=29,
            help="End year"
        )
    
    if start_year >= end_year:
        st.error("⚠️ Start must be before end")
        st.stop()
    
    st.markdown("##### 💰 Initial Capital")
    initial_capital = st.number_input("Amount ($)", 100_000, 1_000_000_000, 10_000_000, 1_000_000, format="%d")
    
    st.markdown("##### 🎛️ STORMPROOF 2.0 Options")
    include_btc = st.checkbox("Include BTC (2014+)", value=True)
    use_dynamic_leverage = st.checkbox("Dynamic Leverage", value=True)
    target_vol = st.slider("Target Vol", 0.05, 0.20, 0.10, 0.01) if use_dynamic_leverage else 0.10
    max_leverage = st.slider("Max Leverage", 1.0, 3.0, 2.0, 0.1) if use_dynamic_leverage else 1.0
    
    st.markdown("---")
    st.success("📦 Local data: Instant loading!")
    run_simulation = st.button("🚀 RUN ANALYSIS", type="primary")

# ========================== LANDING PAGE ==========================
if not run_simulation:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎯 Objective")
        st.markdown("Outperform Bridgewater's All Weather since its **1996 launch**.")
    with col2:
        st.markdown("#### 📊 Universe")
        st.markdown("**SPY** • **TLT** • **GLD** • **DBC** • **BTC** (2014+)")
    with col3:
        st.markdown("#### ⚡ Edge")
        st.markdown("Dynamic leverage, trend overlay, crisis alpha, 4-regime detection.")
    
    st.markdown("---")
    st.markdown("#### 📈 Full Historical Coverage (1996-2025)")
    
    st.markdown("""
    | Period | Market Events Covered |
    |--------|----------------------|
    | **1996-2000** | Asian Crisis, LTCM, Dot-com Bubble |
    | **2001-2007** | 9/11, Iraq War, Housing Boom |
    | **2008-2012** | Financial Crisis, EU Debt Crisis |
    | **2013-2019** | QE Bull Market, China Crash |
    | **2020-2025** | COVID Crash, Inflation, Rate Hikes |
    """)
    
    st.markdown("---")
    st.markdown("#### 🔬 STORMPROOF 2.0 Enhancements")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Bridgewater 2014+ improvements:**
        - Dynamic leverage (0.5x - 2.5x)
        - Trend-following overlay
        - Enhanced 4-regime detection
        - Better inflation signals
        """)
    with col2:
        st.markdown("""
        **Additional alpha:**
        - BTC allocation (0→8% since 2014)
        - Crisis alpha (VIX > 80 → TLT)
        - Quantum Monte-Carlo
        - Pearl Causal Inference
        """)
    
    st.markdown("---")
    st.markdown('<p class="methodology-text">📋 Proprietary methodology — Request full documentation for institutional due diligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
    st.stop()

# ========================== DATA LOADING ==========================
tickers = ['SPY', 'TLT', 'GLD', 'DBC']
data_messages = []

market_df, macro_df, btc_df, data_loaded = load_local_data()

if not data_loaded:
    st.error("❌ Failed to load local data files. Please ensure market_data.csv, macro_data.csv, and btc_data.csv are in the same directory as app.py")
    st.stop()

# Filter by date range
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

df = market_df[(market_df.index >= start_date) & (market_df.index <= end_date)].copy()

# Merge macro data
for col in macro_df.columns:
    if col not in df.columns:
        df = df.join(macro_df[[col]], how='left')
        df[col] = df[col].ffill().bfill()

# Add BTC if enabled and available
btc_available = False
if include_btc and btc_df is not None:
    btc_filtered = btc_df[(btc_df.index >= start_date) & (btc_df.index <= end_date)]
    if len(btc_filtered) > 0:
        df = df.join(btc_filtered, how='left')
        btc_available = True

# Compute derived indicators
df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
df['CPI_YoY_12m_avg'] = df['CPI_YoY'].rolling(12).mean()
df['UNRATE_change'] = df['UNRATE'].diff()

# Compute returns
returns = df[tickers].pct_change()

# Clean data
valid_mask = returns.notna().all(axis=1) & df['CPI_YoY'].notna()
df = df[valid_mask]
returns = returns[valid_mask]

if 'BTC' in df.columns:
    df['BTC_ret'] = df['BTC'].pct_change()

data_messages.append(f"✅ {len(df)} months loaded ({df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')})")
data_messages.append("📦 Local embedded data")

if len(df) < 24:
    st.error("Insufficient data for selected period")
    st.stop()

# ========================== SIMULATION ==========================
capital_stormproof = []
capital_allweather = []
weights_aw = np.array([0.30, 0.55, 0.075, 0.075])
peak = initial_capital
portfolio_returns = []

cap_storm = initial_capital
cap_aw = initial_capital

for idx in range(len(returns)):
    current_date = df.index[idx]
    current_year = current_date.year
    
    # === ALL WEATHER ===
    ret_aw = np.dot(weights_aw, returns.iloc[idx].values)
    cap_aw *= (1 + ret_aw)
    capital_allweather.append(cap_aw)
    
    # === STORMPROOF 2.0 ===
    window = returns.iloc[max(0, idx-36):idx]
    
    if len(window) < 12:
        ret_storm = ret_aw
    else:
        regime = detect_regime(df.iloc[idx])
        vix = df['VIX'].iloc[idx]
        crisis_w = crisis_alpha(vix)
        
        if crisis_w is not None:
            final_w = crisis_w
            leverage = 0.8
        else:
            base_w = get_regime_weights(regime)
            cov = window.cov() * 252 + np.eye(4) * 1e-6
            rp_w = risk_parity(cov)
            
            trend = trend_signal(returns.iloc[max(0, idx-12):idx])
            w_trend = base_w * trend
            w_trend = w_trend / w_trend.sum() if w_trend.sum() > 0 else base_w
            
            recent_rets = returns.iloc[max(0, idx-3):idx].mean().values
            vix_w, _ = vix_panic_buy(vix, base_w, recent_rets)
            
            cpi_change = df['CPI_YoY'].iloc[idx] - df['CPI_YoY'].iloc[idx-1] if idx > 0 else 0
            fed_change = df['FEDFUNDS'].iloc[idx] - df['FEDFUNDS'].iloc[idx-1] if idx > 0 else 0
            causal_adj = causal_adjustment(regime, cpi_change, fed_change)
            
            mc_direction = quantum_monte_carlo(window)
            mc_signal = (mc_direction > 0).astype(float)
            
            tension = elastic_tension(df.iloc[idx])
            final_w = (
                0.35 * rp_w +
                0.25 * w_trend +
                0.20 * vix_w +
                0.10 * (base_w + causal_adj) +
                0.05 * mc_signal +
                0.05 * np.ones(4) * (1 - tension)
            )
            final_w = np.maximum(final_w, 0.03)
            final_w /= final_w.sum()
            
            if use_dynamic_leverage and len(portfolio_returns) >= 12:
                realized_vol = np.std(portfolio_returns[-60:]) * np.sqrt(12) if len(portfolio_returns) >= 60 else np.std(portfolio_returns[-12:]) * np.sqrt(12)
                leverage = dynamic_leverage(realized_vol, target_vol, max_leverage)
            else:
                leverage = 1.0
            
            current_dd = (cap_storm - peak) / peak if peak > 0 else 0
            if current_dd < -0.20:
                leverage = min(leverage, 0.6)
        
        ret_storm = np.dot(final_w, returns.iloc[idx].values) * leverage
        
        # BTC contribution (2014+)
        if include_btc and btc_available and 'BTC' in df.columns and current_year >= 2014:
            if idx >= 12 and 'BTC_ret' in df.columns:
                gold_ret_12m = df['GLD'].iloc[max(0, idx-12):idx].pct_change().sum()
                btc_vals = df['BTC'].iloc[max(0, idx-12):idx].dropna()
                btc_ret_12m = btc_vals.pct_change().sum() if len(btc_vals) > 1 else None
                
                _, btc_alloc = gold_btc_balance(gold_ret_12m, btc_ret_12m, current_year)
                
                if 'BTC_ret' in df.columns and not pd.isna(df['BTC_ret'].iloc[idx]):
                    btc_ret_now = df['BTC_ret'].iloc[idx]
                    if btc_alloc > 0:
                        ret_storm += btc_alloc * btc_ret_now * leverage * 0.5
    
    portfolio_returns.append(ret_storm)
    cap_storm *= (1 + ret_storm)
    capital_stormproof.append(cap_storm)
    peak = max(peak, cap_storm)

# ========================== RESULTS ==========================
result = pd.DataFrame({
    "STORMPROOF 2.0": capital_stormproof,
    "All Weather": capital_allweather
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
st.markdown('<p class="methodology-text">📋 STORMPROOF 2.0 (Dynamic Leverage • Trend Overlay • 4-Regime Detection • BTC/Gold Balance • Crisis Alpha • Quantum Monte-Carlo • Pearl Causal Inference) — Request full documentation for institutional due diligence</p>', unsafe_allow_html=True)
st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)
