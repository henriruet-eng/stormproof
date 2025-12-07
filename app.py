# ======================================================================
# STORMPROOF 2025 — Institutional Bot-Advisor
# Risk Parity Backtesting vs Ray Dalio's All Weather (since 1996)
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
    page_title="STORMPROOF - Institutional Robo-Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode CSS for institutional clients
st.markdown("""
<style>
    /* Global dark mode */
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
    
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d0d12;
        border-right: 1px solid #1a1a2e;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #c0c0c0;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Buttons */
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
    
    /* DataFrames */
    .stDataFrame {
        background-color: #12121a;
        border-radius: 8px;
    }
    
    /* Footer styles */
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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Info boxes */
    .stAlert {
        background-color: #12121a;
        border: 1px solid #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ========================== HISTORICAL MARKET CRISES ==========================
MARKET_CRISES = [
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
    """Download data from Yahoo Finance API"""
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
    """Download data from FRED (Federal Reserve Economic Data)"""
    try:
        data = pdr.DataReader(series_id, 'fred', start_date, end_date)
        return data
    except Exception:
        return None


@st.cache_data(ttl=7200, show_spinner=False)
def get_historical_data(start_year, end_year):
    """
    Get historical data using ETFs when available, proxies for earlier periods.
    
    Data sources:
    - Equities: SPY (1993+) or ^GSPC (S&P 500 Index)
    - Bonds: TLT (2002+) or calculated from FRED Treasury yields
    - Gold: GLD (2004+) or GC=F (Gold Futures) or FRED gold prices
    - Commodities: DBC (2006+) or ^GSCI (S&P GSCI) or simulated from components
    - VIX: ^VIX (1990+)
    - Macro: FRED (CPI, Fed Funds, Unemployment)
    """
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    data_sources = {}
    series_list = []
    
    # ===== EQUITIES =====
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
    
    # ===== LONG-TERM BONDS =====
    tlt_data = download_yahoo_data('TLT', start_date, end_date)
    if tlt_data is not None and len(tlt_data) > 0:
        tlt_data.name = 'TLT'
        series_list.append(tlt_data)
        data_sources['TLT'] = 'Yahoo Finance (TLT ETF)'
    else:
        ief_data = download_yahoo_data('IEF', start_date, end_date)
        if ief_data is not None:
            ief_data.name = 'TLT'
            series_list.append(ief_data)
            data_sources['TLT'] = 'Yahoo Finance (IEF proxy)'
        else:
            data_sources['TLT'] = 'Simulated from interest rates'
    
    # ===== GOLD =====
    gld_data = download_yahoo_data('GLD', start_date, end_date)
    if gld_data is not None and len(gld_data) > 0:
        gld_data.name = 'GLD'
        series_list.append(gld_data)
        data_sources['GLD'] = 'Yahoo Finance (GLD ETF)'
    else:
        gc_data = download_yahoo_data('GC=F', start_date, end_date)
        if gc_data is not None:
            gc_data.name = 'GLD'
            series_list.append(gc_data)
            data_sources['GLD'] = 'Yahoo Finance (Gold Futures proxy)'
    
    # ===== COMMODITIES =====
    dbc_data = download_yahoo_data('DBC', start_date, end_date)
    if dbc_data is not None and len(dbc_data) > 0:
        dbc_data.name = 'DBC'
        series_list.append(dbc_data)
        data_sources['DBC'] = 'Yahoo Finance (DBC ETF)'
    else:
        gsci_data = download_yahoo_data('^SPGSCI', start_date, end_date)
        if gsci_data is not None:
            gsci_data.name = 'DBC'
            series_list.append(gsci_data)
            data_sources['DBC'] = 'Yahoo Finance (S&P GSCI proxy)'
        else:
            gsg_data = download_yahoo_data('GSG', start_date, end_date)
            if gsg_data is not None:
                gsg_data.name = 'DBC'
                series_list.append(gsg_data)
                data_sources['DBC'] = 'Yahoo Finance (GSG proxy)'
    
    # ===== VIX =====
    vix_data = download_yahoo_data('^VIX', start_date, end_date)
    if vix_data is not None:
        vix_data.name = '^VIX'
        series_list.append(vix_data)
        data_sources['VIX'] = 'Yahoo Finance (CBOE VIX Index)'
    
    return series_list, data_sources


@st.cache_data(ttl=7200, show_spinner=False)
def get_fred_macro_data(start_date, end_date):
    """Get macroeconomic data from FRED"""
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
        
        return macro_data, sources
    except Exception as e:
        return None, str(e)


# ========================== STRATEGY FUNCTIONS ==========================

def detect_dalio_season(row):
    """Detect economic season per Ray Dalio's framework"""
    growth = row['UNRATE'] < 5.5
    inflation = row['CPI_YoY'] > 2.5
    if growth and not inflation:
        return "Spring"
    elif growth and inflation:
        return "Summer"
    elif not growth and not inflation:
        return "Fall"
    return "Winter"


def elastic_tension(row):
    """Calculate market elastic tension"""
    tension = 0
    if row['^VIX'] > 40:
        tension += 0.4
    if abs(row['CPI_YoY']) > 4:
        tension += 0.3
    if row['FEDFUNDS'] < 1:
        tension += 0.2
    return min(tension, 1.0)


def vix_panic_buy(vix, current_w, recent_returns):
    """Aggressive buying during VIX spikes (mean reversion)"""
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


def causal_adjustment(season, cpi_change, fed_change):
    """Causal inference adjustments based on macro regime"""
    adj = np.zeros(4)
    if "Summer" in season or "Winter" in season:
        adj[2] += 0.15
    if "Winter" in season and fed_change < -0.5:
        adj[1] += 0.25
    if cpi_change > 1:
        adj[3] += 0.10
    return adj


def quantum_monte_carlo(returns_window, n_sim=1000):
    """Monte-Carlo optimization with quantum-inspired selection"""
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
    """Dynamic Risk Parity allocation"""
    vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    w = 1 / vol
    return w / w.sum()


def double_loop_feedback(cumulative_drawdown):
    """Introspective feedback loop for risk adjustment"""
    return 1.2 if cumulative_drawdown < -0.15 else 1.0


# ========================== HEADER ==========================
st.markdown('<h1 class="main-header">⚡ STORMPROOF</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Institutional Robo-Advisor • Risk Parity vs All Weather Backtesting</p>', unsafe_allow_html=True)

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
            index=29,
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
    
    run_simulation = st.button("🚀 RUN ANALYSIS", type="primary")

# ========================== MAIN CONTENT ==========================

if not run_simulation:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Objective")
        st.markdown("Optimize risk-adjusted returns compared to Bridgewater's All Weather strategy (launched 1996).")
    
    with col2:
        st.markdown("#### 📊 Investment Universe")
        st.markdown("**Equities** (SPY) • **Long Bonds** (TLT) • **Gold** (GLD) • **Commodities** (DBC)")
    
    with col3:
        st.markdown("#### ⚡ Competitive Edge")
        st.markdown("Dynamic allocation based on macroeconomic regimes and implied volatility.")
    
    st.markdown("---")
    
    st.markdown("#### 📈 Data Sources")
    st.markdown("""
    - **Market Data**: Yahoo Finance API (ETFs & Index proxies for pre-ETF periods)
    - **Macro Data**: Federal Reserve Economic Data (FRED) — CPI, Fed Funds Rate, Unemployment
    - **Volatility**: CBOE VIX Index (since 1990)
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
source_info = []

with st.spinner('Loading market data...'):
    try:
        series_list, data_sources = get_historical_data(start_year, end_year)
        
        if len(series_list) < 4:
            st.error(f"❌ Insufficient data: only {len(series_list)} assets loaded. Need 4.")
            st.stop()
        
        for asset, source in data_sources.items():
            source_info.append(f"{asset}: {source}")
        
        prices = pd.concat(series_list, axis=1).resample('M').last()
        df = prices.copy()
        
        for ticker in tickers + ['^VIX']:
            if ticker not in df.columns:
                st.error(f"❌ Missing data for {ticker}")
                st.stop()
        
        fred_loaded = False
        if FRED_AVAILABLE:
            macro_data, macro_sources = get_fred_macro_data(start_date, end_date)
            
            if macro_data and len(macro_data) >= 3:
                for key, series in macro_data.items():
                    monthly = series.resample('M').last()
                    df = df.join(monthly, how='left')
                    df[key] = df[key].ffill().bfill()
                
                fred_loaded = True
                data_messages.append("✅ FRED macro data loaded (CPI, Fed Funds, Unemployment)")
                
                if isinstance(macro_sources, dict):
                    for name, src in macro_sources.items():
                        source_info.append(f"{name}: {src}")
        
        if not fred_loaded:
            np.random.seed(42)
            n = len(df)
            df['CPIAUCSL'] = 150 + np.cumsum(np.random.normal(0.3, 0.4, n))
            df['FEDFUNDS'] = np.clip(4.0 + np.cumsum(np.random.normal(0, 0.3, n)), 0, 10)
            df['UNRATE'] = np.clip(5.5 + np.cumsum(np.random.normal(0, 0.15, n)), 3, 12)
            data_messages.append("⚠️ FRED unavailable — Simulated macro data used")
        
        df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
        
        returns_raw = df[tickers].pct_change()
        valid_mask = returns_raw.notna().all(axis=1) & df['CPI_YoY'].notna()
        df = df[valid_mask]
        returns = returns_raw[valid_mask]
        
        data_messages.append(f"✅ {len(df)} months loaded ({df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')})")
        
        if len(df) < 24:
            st.error(f"Insufficient data ({len(df)} months). Minimum required: 24 months.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ========================== SIMULATION ==========================

df['Dalio_Season'] = df.apply(detect_dalio_season, axis=1)
df['Elastic_Tension'] = df.apply(elastic_tension, axis=1)

with st.spinner('Running simulation...'):
    capital_stormproof = []
    capital_allweather = []
    weights_aw = np.array([0.30, 0.55, 0.075, 0.075])
    cumulative_dd = 0
    peak_storm = initial_capital
    
    cap_storm = initial_capital
    cap_aw = initial_capital
    
    for idx in range(len(returns)):
        window_start = max(0, idx - 36)
        window = returns.iloc[window_start:idx]
        
        if len(window) < 12:
            ret_storm = np.dot(weights_aw, returns.iloc[idx].values)
            ret_aw = np.dot(weights_aw, returns.iloc[idx].values)
        else:
            season = df['Dalio_Season'].iloc[idx]
            cpi_change = df['CPI_YoY'].iloc[idx] - df['CPI_YoY'].iloc[idx-1] if idx > 0 else 0
            fed_change = df['FEDFUNDS'].iloc[idx] - df['FEDFUNDS'].iloc[idx-1] if idx > 0 else 0
            causal_adj = causal_adjustment(season, cpi_change, fed_change)
            
            vix = df['^VIX'].iloc[idx]
            tension = df['Elastic_Tension'].iloc[idx]
            
            recent_30d = returns.iloc[max(0, idx-30):idx].mean().values
            vix_w, _ = vix_panic_buy(vix, weights_aw, recent_30d)
            
            cov = window.cov() * 252 + np.eye(4) * 1e-6
            rp_w = risk_parity(cov)
            mc_direction = quantum_monte_carlo(window)
            
            losses = np.where(mc_direction < 0, mc_direction * 2.2, mc_direction)
            mc_adjusted = mc_direction - losses.mean() * 0.05
            
            final_w = (
                0.40 * rp_w +
                0.25 * vix_w +
                0.15 * (weights_aw + causal_adj) +
                0.15 * (mc_adjusted > 0).astype(float) +
                0.05 * np.ones(4) * (1 - tension)
            )
            final_w = np.maximum(final_w, 0)
            final_w /= final_w.sum()
            
            peak_storm = max(peak_storm, cap_storm)
            dd = (cap_storm - peak_storm) / peak_storm if peak_storm > 0 else 0
            cumulative_dd = min(cumulative_dd, dd)
            final_w *= double_loop_feedback(cumulative_dd)
            final_w /= final_w.sum()
            
            ret_storm = np.dot(final_w, returns.iloc[idx].values)
            ret_aw = np.dot(weights_aw, returns.iloc[idx].values)
        
        cap_storm *= (1 + ret_storm)
        cap_aw *= (1 + ret_aw)
        
        capital_stormproof.append(cap_storm)
        capital_allweather.append(cap_aw)

# ========================== RESULTS ==========================

result = pd.DataFrame({
    "STORMPROOF": capital_stormproof,
    "All Weather": capital_allweather
}, index=returns.index)

final_storm = result["STORMPROOF"].iloc[-1]
final_aw = result["All Weather"].iloc[-1]
years = (result.index[-1] - result.index[0]).days / 365.25
annual_return_storm = ((final_storm / initial_capital) ** (1 / years) - 1) * 100
annual_return_aw = ((final_aw / initial_capital) ** (1 / years) - 1) * 100

# ========================== CHART (TOP) ==========================

st.markdown("### 📈 Comparative Performance")

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')

result.plot(ax=ax, linewidth=2.5, color=['#667eea', '#f093fb'], alpha=0.95)

for crisis in MARKET_CRISES:
    crisis_start = pd.to_datetime(crisis["start"])
    crisis_end = pd.to_datetime(crisis["end"])
    
    if crisis_start >= result.index[0] and crisis_start <= result.index[-1]:
        ax.axvspan(crisis_start, min(crisis_end, result.index[-1]), 
                   alpha=0.12, color='#ff4444', zorder=0)
        
        mid_date = crisis_start + (min(crisis_end, result.index[-1]) - crisis_start) / 2
        
        try:
            y_val = result.loc[result.index >= crisis_start].iloc[0].max()
            y_pos = y_val * 1.08
        except:
            y_pos = result.max().max() * 0.9
        
        ax.annotate(
            f"{crisis['name']}\n{crisis['duration']}",
            xy=(mid_date, y_pos),
            fontsize=7,
            color='#ff6666',
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ff4444', alpha=0.9),
            zorder=5
        )

ax.set_title(
    f"STORMPROOF vs All Weather ({start_year} → {end_year})",
    fontsize=20,
    fontweight='bold',
    color='#e0e0e0',
    pad=15
)
ax.set_ylabel("Portfolio Value ($)", fontsize=13, color='#a0a0a0')
ax.set_xlabel("")
ax.grid(alpha=0.1, linestyle='-', color='#404050')
ax.legend(fontsize=12, loc='upper left', facecolor='#12121a', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
ax.tick_params(colors='#a0a0a0', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#2a2a3a')
ax.spines['left'].set_color('#2a2a3a')

def format_millions(x, pos):
    if x >= 1e9:
        return f'{x/1e9:.1f}B$'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M$'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K$'
    return f'{x:.0f}$'

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_millions))

plt.tight_layout()
st.pyplot(fig)

# ========================== METRICS (BELOW CHART) ==========================

st.markdown("---")
st.markdown("### 📊 Simulation Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 STORMPROOF Final",
        f"{final_storm/1e6:.2f}M $",
        f"+{((final_storm / initial_capital - 1) * 100):.1f}%"
    )

with col2:
    st.metric(
        "📈 All Weather Final",
        f"{final_aw/1e6:.2f}M $",
        f"+{((final_aw / initial_capital - 1) * 100):.1f}%"
    )

with col3:
    surperf = (final_storm / final_aw - 1) * 100
    st.metric(
        "🚀 Outperformance",
        f"+{surperf:.1f}%",
        "vs All Weather"
    )

with col4:
    st.metric(
        "📈 Annualized Return",
        f"{annual_return_storm:.2f}%",
        "STORMPROOF"
    )

# ========================== RISK METRICS ==========================

st.markdown("---")
st.markdown("### 📉 Risk Metrics")

vol_storm = result["STORMPROOF"].pct_change().std() * np.sqrt(12) * 100
vol_aw = result["All Weather"].pct_change().std() * np.sqrt(12) * 100
dd_storm = ((result["STORMPROOF"] / result["STORMPROOF"].cummax()) - 1).min() * 100
dd_aw = ((result["All Weather"] / result["All Weather"].cummax()) - 1).min() * 100

rf_rate = df['FEDFUNDS'].mean() if 'FEDFUNDS' in df.columns else 2.0
sharpe_storm = (annual_return_storm - rf_rate) / vol_storm if vol_storm > 0 else 0
sharpe_aw = (annual_return_aw - rf_rate) / vol_aw if vol_aw > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    stats_storm = pd.DataFrame({
        'STORMPROOF': [
            f"{annual_return_storm:.2f}%",
            f"{vol_storm:.2f}%",
            f"{dd_storm:.2f}%",
            f"{sharpe_storm:.2f}"
        ]
    }, index=['Annualized Return', 'Volatility', 'Max Drawdown', 'Sharpe Ratio'])
    st.dataframe(stats_storm, use_container_width=True)

with col2:
    stats_aw = pd.DataFrame({
        'All Weather': [
            f"{annual_return_aw:.2f}%",
            f"{vol_aw:.2f}%",
            f"{dd_aw:.2f}%",
            f"{sharpe_aw:.2f}"
        ]
    }, index=['Annualized Return', 'Volatility', 'Max Drawdown', 'Sharpe Ratio'])
    st.dataframe(stats_aw, use_container_width=True)

with col3:
    stats_diff = pd.DataFrame({
        'Difference': [
            f"{annual_return_storm - annual_return_aw:+.2f}%",
            f"{vol_storm - vol_aw:+.2f}%",
            f"{dd_storm - dd_aw:+.2f}%",
            f"{sharpe_storm - sharpe_aw:+.2f}"
        ]
    }, index=['Annualized Return', 'Volatility', 'Max Drawdown', 'Sharpe Ratio'])
    st.dataframe(stats_diff, use_container_width=True)

# ========================== FOOTER ==========================

st.markdown("---")

footer_data = " • ".join(data_messages)
st.markdown(f'<p class="footer-text">{footer_data}</p>', unsafe_allow_html=True)

st.markdown('<p class="methodology-text">📋 Proprietary methodology (Dalio 4 Seasons • Pearl Causal Inference • Quantum Monte-Carlo • Elastic Networks • Six Thinking Hats • Double-Loop Learning • Prospect Theory • VIX Panic Buying • Dynamic Risk Parity) — Request full documentation for institutional due diligence</p>', unsafe_allow_html=True)

st.markdown('<p class="footer-contact">Contact: henri8@gmail.com • +33 6 63 54 7000</p>', unsafe_allow_html=True)



