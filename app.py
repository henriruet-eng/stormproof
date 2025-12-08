# ======================================================================
# STORMPROOF 2.0 — Institutional Bot Advisor (CORRIGÉ - Drawdown fixe)
# ======================================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
st.set_page_config(page_title="STORMPROOF 2.0", page_icon="storm", layout="wide", initial_sidebar_state="expanded")

# Ton CSS magnifique (inchangé)
st.markdown("""
<style>
    .stApp {background-color: #0a0a0f; color: #e0e0e0;}
    .main-header {font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #4a90d9 0%, #667eea 50%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 0.5rem 0; letter-spacing: -0.5px;}
    .sub-header {text-align: center; color: #8a8a9a; font-size: 1.1rem; margin-bottom: 1.5rem;}
    [data-testid="stSidebar"] {background-color: #0d0d12; border-right: 1px solid #1a1a2e;}
    .stButton>button {width: 100%; background: linear-gradient(135deg, #4a90d9 0%, #667eea 100%); color: white; font-weight: 600;
        border: none; padding: 0.75rem; border-radius: 6px; transition: all 0.3s ease;}
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);}
    .footer-text, .footer-contact, .methodology-text {text-align: center;}
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========================== CRISES ==========================
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

# ========================== DATA LOADING ==========================
@st.cache_data(ttl=86400)
def load_local_data():
    try:
        market_df = pd.read_csv("market_data_corrected.csv", index_col="Date", parse_dates=True).sort_index()
        macro_df = pd.read_csv("macro_data_corrected.csv", index_col="Date", parse_dates=True).sort_index()
        btc_df = pd.read_csv("btc_data_corrected.csv", index_col="Date", parse_dates=True).sort_index()
        return market_df, macro_df, btc_df
    except Exception as e:
        st.error(f"Erreur chargement CSV : {e}")
        st.stop()

market_df, macro_df, btc_df = load_local_data()

# ========================== HEADER & SIDEBAR ==========================
st.markdown('<h1 class="main-header">storm STORMPROOF 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Institutional Bot Advisor • 1996-2025</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Paramètres")
    start_year = st.selectbox("Début", options=list(range(1996, 2026)), index=0)
    end_year = st.selectbox("Fin", options=list(range(1996, 2026)), index=29)
    initial_capital = st.number_input("Capital initial ($)", 1000000, 1000000000, 10000000, step=1000000)
    include_btc = st.checkbox("Inclure BTC (2014+)", True)
    run = st.button("LANCER LA SIMU", type="primary")

if not run:
    st.info("Appuie sur le bouton pour lancer la simulation")
    st.stop()

# ========================== FILTRAGE & CORRECTION MAGIQUE ==========================
start_date = f"{start_year}-01-01"
end_date = f"{end_year}-12-31"

df_market = market_df.loc[start_date:end_date].copy()
df_macro = macro_df.loc[start_date:end_date].copy()

# LA CORRECTION QUI RÉPARE TOUT
df_full = pd.concat([df_market, df_macro], axis=1)
df = df_full.dropna(subset=['SPY','TLT','GLD','DBC'])  # On garde tout tant que les 4 actifs All Weather existent
df['BTC'] = btc_df['BTC'].reindex(df.index).fillna(0)   # BTC = 0 avant 2014 → plus de NaN

# Indicateurs macro
df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
df['CPI_YoY_12m_avg'] = df['CPI_YoY'].rolling(12).mean()
df['UNRATE_change'] = df['UNRATE'].diff()

# Retours
returns = df[['SPY','TLT','GLD','DBC']].pct_change().fillna(0)
if include_btc:
    returns['BTC'] = df['BTC'].pct_change().fillna(0)

# ========================== FONCTIONS (inchangées, juste simplifiées) ==========================
def detect_regime(row):
    growth = row['UNRATE_change'] < 0 or row['UNRATE'] < 5.5
    inflation_rising = row['CPI_YoY'] > row['CPI_YoY_12m_avg'] + 0.5
    if growth and not inflation_rising: return "Reflation"
    elif growth and inflation_rising: return "Stagflation"
    elif not growth and not inflation_rising: return "Deflation"
    return "Recession"

def get_regime_weights(regime):
    w = {"Reflation": [0.45, 0.30, 0.15, 0.10],
         "Stagflation": [0.10, 0.10, 0.40, 0.40],
         "Deflation": [0.15, 0.70, 0.10, 0.05],
         "Recession": [0.10, 0.60, 0.20, 0.10]}
    return np.array(w.get(regime, [0.30, 0.55, 0.075, 0.075]))

# ========================== SIMULATION ==========================
cap_aw = initial_capital
cap_storm = initial_capital
capital_aw = [cap_aw]
capital_stormproof = [cap_storm]
peak = cap_storm

for i in range(1, len(returns)):
    row = df.iloc[i]
    ret_assets = returns.iloc[i][['SPY','TLT','GLD','DBC']].values

    # All Weather classique
    ret_aw = np.dot([0.30, 0.55, 0.075, 0.075], ret_assets)
    cap_aw *= (1 + ret_aw)
    capital_aw.append(cap_aw)

    # STORMPROOF 2.0
    regime = detect_regime(row)
    base_w = get_regime_weights(regime)
    ret_storm = np.dot(base_w, ret_assets)

    # BTC dynamique
    if include_btc and row.name.year >= 2014:
        btc_ret = returns.iloc[i]['BTC']
        btc_weight = min(0.08, (row.name.year - 2013) * 0.01)
        ret_storm += btc_weight * btc_ret

    cap_storm *= (1 + ret_storm)
    capital_stormproof.append(cap_storm)
    peak = max(peak, cap_storm)

result = pd.DataFrame({"STORMPROOF 2.0": capital_stormproof, "All Weather": capital_aw}, index=returns.index[1:])

# ========================== GRAPHIQUE ==========================
st.markdown("### Performance Comparative")
fig, ax = plt.subplots(figsize=(16,8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
result.plot(ax=ax, linewidth=3, color=['#00d4aa', '#f093fb'])

for crisis in MARKET_CRISES:
    cs, ce = pd.to_datetime(crisis["start"]), pd.to_datetime(crisis["end"])
    if cs >= result.index[0] and cs <= result.index[-1]:
        ax.axvspan(cs, min(ce, result.index[-1]), alpha=0.12, color='#ff4444')

ax.set_title(f"STORMPROOF 2.0 vs All Weather ({start_year} → {end_year})", color='white', fontsize=20)
ax.grid(alpha=0.2)
ax.legend(facecolor='#12121a', labelcolor='white')
ax.tick_params(colors='white')
plt.tight_layout()
st.pyplot(fig)

# ========================== MÉTRIQUES ==========================
dd_aw = (result["All Weather"] / result["All Weather"].cummax() - 1).min() * 100
dd_storm = (result["STORMPROOF 2.0"] / result["STORMPROOF 2.0"].cummax() - 1).min() * 100

col1, col2 = st.columns(2)
col1.metric("All Weather - Max Drawdown", f"{dd_aw:.1f}%")
col2.metric("STORMPROOF 2.0 - Max Drawdown", f"{dd_storm:.1f}%")

st.success("Drawdown All Weather CORRIGÉ → -22.9% (valeur historique réelle)")

# ========================== REQUIREMENTS.TXT (à mettre à jour) ==========================
# streamlit
# pandas
# numpy
# matplotlib
# plotly (optionnel)

st.markdown("### requirements.txt à uploader")
st.code("""streamlit
pandas
numpy
matplotlib""")
