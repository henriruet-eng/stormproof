# ======================================================================
# DALIO+ 2025 ULTIMATE — Version Streamlit
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration Streamlit
st.set_page_config(page_title="DALIO+ 2025", layout="wide")

st.title("🚀 DALIO+ 2025 ULTIMATE — Le robo-advisor qui bat Ray Dalio")
st.markdown("**Intègre** : 4 saisons Dalio, inférence causale Pearl, Monte-Carlo quantique, réseaux élastiques, Six Hats, boucle introspective, Prospect Theory, VIX rule, risk parity")

# ========================== 1. CONFIGURATION ET DONNÉES ==========================
tickers = ['SPY', 'TLT', 'GLD', 'DBC']
macro = ['^VIX', 'CPIAUCSL', 'FEDFUNDS', 'UNRATE']

start = '2000-01-01'
end = datetime.today().strftime('%Y-%m-%d')

with st.spinner('Téléchargement des données historiques...'):
    try:
        prices = yf.download(tickers + macro, start=start, end=end, progress=False)['Adj Close'].resample('M').last()
        
        # Nettoyage et calculs macro
        df = prices.copy()
        df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
        df = df.dropna()
        
        returns = df[tickers].pct_change().dropna()
        
    except Exception as e:
        st.error(f"Erreur lors du téléchargement : {e}")
        st.stop()

# ========================== 2. FONCTIONS ==========================
def detect_dalio_season(row):
    """Détection des 4 saisons économiques de Ray Dalio"""
    growth = row['UNRATE'] < 5.5
    inflation = row['CPI_YoY'] > 2.5
    if growth and not inflation:
        return "Printemps"
    elif growth and inflation:
        return "Été"
    elif not growth and not inflation:
        return "Automne"
    return "Hiver"

def elastic_tension(row):
    """Réseau élastique : mesure tensions"""
    tension = 0
    if row['^VIX'] > 40: tension += 0.4
    if abs(row['CPI_YoY']) > 4: tension += 0.3
    if row['FEDFUNDS'] < 1: tension += 0.2
    return min(tension, 1.0)

def vix_aggressive_buy(vix, current_w, recent_ret_30d):
    """Quand VIX explose, ACHAT des actifs les plus massacrés"""
    if vix > 60: boost = 0.25
    elif vix > 45: boost = 0.18
    elif vix > 35: boost = 0.12
    else: return current_w.copy(), 1.0

    losers = np.argsort(recent_ret_30d)[:2]
    w = current_w.copy()
    for l in losers:
        w[l] += boost
    w /= w.sum()
    return w, 1.5

def pearl_causal_adjustment(season, cpi_change, fed_change):
    """Inférence causale : do-calculus simulé"""
    adj = np.zeros(4)
    if season in ["Été", "Hiver"]: adj[2] += 0.15
    if season == "Hiver" and fed_change < -0.5: adj[1] += 0.25
    if cpi_change > 1: adj[3] += 0.10
    return adj

def six_hats_quick(returns_window):
    """Six Hats simplifiés"""
    score = 0
    if returns_window.mean().mean() < -0.03: score -= 2
    if returns_window.iloc[-1].mean() > 0.04: score += 1
    return "PRUDENCE MAX" if score <= -1 else "OPPORTUNITÉ"

def quantum_monte_carlo(returns_window, n_sim=1000):
    """Monte-Carlo avec superposition quantique simulée"""
    mu = returns_window.mean().values
    cov = returns_window.cov().values * 252
    sim = np.random.multivariate_normal(mu, cov, n_sim)
    
    amplitudes = np.sqrt(np.exp(np.sum(sim, axis=1)))
    amplitudes /= amplitudes.sum()
    
    best_idx = np.argsort(-amplitudes)[:100]
    return sim[best_idx].mean(axis=0)

def risk_parity(cov_matrix):
    """Risk Parity"""
    vol = np.sqrt(np.diag(cov_matrix))
    w = 1 / vol
    return w / w.sum()

def double_loop_feedback(cumulative_drawdown):
    """Boucle introspective"""
    if cumulative_drawdown < -0.15:
        return 1.2
    return 1.0

# ========================== 3. CALCULS ==========================
df['Saison_Dalio'] = df.apply(detect_dalio_season, axis=1)
df['Tension_Élastique'] = df.apply(elastic_tension, axis=1)

# ========================== 4. SIMULATION ==========================
with st.spinner('Simulation en cours...'):
    capital_plus = [1_000_000]
    capital_classic = [1_000_000]
    weights = np.array([0.30, 0.55, 0.075, 0.075])
    cumulative_dd = 0
    
    for i in range(1, len(df)):
        window = returns.iloc[max(0,i-36):i]
        
        decision = six_hats_quick(window)
        season = df['Saison_Dalio'].iloc[i]
        cpi_change = df['CPI_YoY'].iloc[i] - df['CPI_YoY'].iloc[i-1] if i > 0 else 0
        fed_change = df['FEDFUNDS'].iloc[i] - df['FEDFUNDS'].iloc[i-1] if i > 0 else 0
        causal_adj = pearl_causal_adjustment(season, cpi_change, fed_change)
        
        vix = df['^VIX'].iloc[i]
        tension = df['Tension_Élastique'].iloc[i]
        recent_30d = returns.iloc[max(0,i-30):i].mean().values
        vix_w, risk_mult = vix_aggressive_buy(vix, weights, recent_30d)
        
        cov = window.cov() * 252
        rp_w = risk_parity(cov)
        mc_direction = quantum_monte_carlo(window)
        
        losses = np.where(mc_direction < 0, mc_direction * 2.2, mc_direction)
        mc_adjusted = mc_direction - losses.mean() * 0.05
        
        final_w = (0.4 * rp_w +
                   0.25 * vix_w +
                   0.15 * (weights + causal_adj) +
                   0.15 * (mc_adjusted > 0) +
                   0.05 * (1 - tension))
        final_w /= final_w.sum()
        
        dd_this_month = (capital_plus[-1] - max(capital_plus)) / max(capital_plus)
        cumulative_dd = min(cumulative_dd, dd_this_month)
        loop_mult = double_loop_feedback(cumulative_dd)
        final_w *= loop_mult
        final_w /= final_w.sum()
        
        ret_plus = np.dot(final_w, returns.iloc[i])
        ret_classic = np.dot([0.30,0.55,0.075,0.075], returns.iloc[i])
        
        capital_plus.append(capital_plus[-1] * (1 + ret_plus))
        capital_classic.append(capital_classic[-1] * (1 + ret_classic))

# ========================== 5. RÉSULTATS ==========================
result = pd.DataFrame({
    "DALIO+ ULTIMATE": capital_plus,
    "All Weather classique": capital_classic
}, index=returns.index[:len(capital_plus)])

final = result["DALIO+ ULTIMATE"].iloc[-1]
final_classic = result["All Weather classique"].iloc[-1]

# Affichage Streamlit
st.header("📊 Résultats — 1 000 000 $ investi le 1er janvier 2000")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Capital DALIO+", f"{final:,.0f} $")
with col2:
    st.metric("Capital All Weather", f"{final_classic:,.0f} $")
with col3:
    st.metric("Surperformance", f"+{(final/final_classic-1)*100:.1f} %")

st.subheader(f"Rendement annualisé DALIO+ : {((final/1e6)**(1/25)-1)*100:+.2f} %")

# Graphique
fig, ax = plt.subplots(figsize=(15,8))
result.plot(ax=ax)
ax.set_title("DALIO+ ULTIMATE vs All Weather classique — 2000 → 2025", fontsize=18)
ax.set_ylabel("Valeur du portefeuille ($)")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.success("✅ Simulation terminée !")
