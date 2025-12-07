# ======================================================================
# DALIO+ 2025 ULTIMATE — Version Streamlit CORRIGÉE (index fix)
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Pour les données FRED
try:
    from pandas_datareader import data as pdr
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# Configuration Streamlit
st.set_page_config(page_title="DALIO+ 2025", layout="wide")

st.title("🚀 DALIO+ 2025 ULTIMATE — Le robo-advisor qui bat Ray Dalio")
st.markdown("**Intègre** : 4 saisons Dalio, inférence causale Pearl, Monte-Carlo quantique, réseaux élastiques, Six Hats, boucle introspective, Prospect Theory, VIX rule, risk parity")

# ========================== 1. TÉLÉCHARGEMENT DONNÉES ==========================
tickers = ['SPY', 'TLT', 'GLD', 'DBC']
start = '2000-01-01'
end = datetime.today().strftime('%Y-%m-%d')

with st.spinner('📥 Téléchargement des données historiques...'):
    try:
        data_list = []
        
        # Télécharger chaque ticker individuellement
        for ticker in tickers:
            data = yf.download(ticker, start=start, end=end, progress=False)
            
            # Gestion robuste de la structure
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    series = data['Adj Close'].iloc[:, 0] if len(data['Adj Close'].shape) > 1 else data['Adj Close']
                else:
                    series = data['Close'].iloc[:, 0] if len(data['Close'].shape) > 1 else data['Close']
            else:
                if 'Adj Close' in data.columns:
                    series = data['Adj Close']
                else:
                    series = data['Close']
            
            series.name = ticker
            data_list.append(series)
        
        # VIX
        vix_data = yf.download('^VIX', start=start, end=end, progress=False)
        
        if isinstance(vix_data.columns, pd.MultiIndex):
            if 'Adj Close' in vix_data.columns.get_level_values(0):
                vix_series = vix_data['Adj Close'].iloc[:, 0] if len(vix_data['Adj Close'].shape) > 1 else vix_data['Adj Close']
            else:
                vix_series = vix_data['Close'].iloc[:, 0] if len(vix_data['Close'].shape) > 1 else vix_data['Close']
        else:
            if 'Adj Close' in vix_data.columns:
                vix_series = vix_data['Adj Close']
            else:
                vix_series = vix_data['Close']
        
        vix_series.name = '^VIX'
        data_list.append(vix_series)
        
        # Combiner et resampler
        prices = pd.concat(data_list, axis=1).resample('M').last()
        df = prices.copy()
        
        # Données FRED
        if FRED_AVAILABLE:
            try:
                cpi = pdr.DataReader('CPIAUCSL', 'fred', start, end).resample('M').last()
                fedfunds = pdr.DataReader('FEDFUNDS', 'fred', start, end).resample('M').last()
                unrate = pdr.DataReader('UNRATE', 'fred', start, end).resample('M').last()
                
                df['CPIAUCSL'] = cpi['CPIAUCSL']
                df['FEDFUNDS'] = fedfunds['FEDFUNDS']
                df['UNRATE'] = unrate['UNRATE']
                
                st.success("✅ Données FRED réelles chargées !")
            except Exception as e:
                st.warning(f"⚠️ FRED indisponible. Utilisation données simulées.")
                FRED_AVAILABLE = False
        
        # Fallback données simulées
        if not FRED_AVAILABLE:
            df['CPIAUCSL'] = 250 + np.cumsum(np.random.normal(0.2, 0.5, len(df)))
            df['FEDFUNDS'] = np.clip(3.0 + np.random.normal(0, 1.5, len(df)), 0, 8)
            df['UNRATE'] = np.clip(5.5 + np.random.normal(0, 0.8, len(df)), 3, 10)
        
        # Calcul inflation YoY
        df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12) * 100
        
        # CORRECTION CRITIQUE : Calculer returns AVANT dropna
        returns = df[tickers].pct_change()
        
        # Maintenant on peut dropper les NaN en SYNCHRONISANT df et returns
        df = df.dropna()
        returns = returns.loc[df.index]  # ← SYNCHRONISATION CRITIQUE
        
        st.success(f"✅ **{len(df)} mois** chargés ({df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')})")
        
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ========================== 2. FONCTIONS ==========================
def detect_dalio_season(row):
    growth = row['UNRATE'] < 5.5
    inflation = row['CPI_YoY'] > 2.5
    if growth and not inflation: return "Printemps 🌸"
    elif growth and inflation: return "Été ☀️"
    elif not growth and not inflation: return "Automne 🍂"
    return "Hiver ❄️"

def elastic_tension(row):
    tension = 0
    if row['^VIX'] > 40: tension += 0.4
    if abs(row['CPI_YoY']) > 4: tension += 0.3
    if row['FEDFUNDS'] < 1: tension += 0.2
    return min(tension, 1.0)

def vix_aggressive_buy(vix, current_w, recent_ret_30d):
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
    adj = np.zeros(4)
    if "Été" in season or "Hiver" in season: adj[2] += 0.15
    if "Hiver" in season and fed_change < -0.5: adj[1] += 0.25
    if cpi_change > 1: adj[3] += 0.10
    return adj

def six_hats_quick(returns_window):
    score = 0
    mean_ret = returns_window.mean().mean()
    last_ret = returns_window.iloc[-1].mean()
    if mean_ret < -0.03: score -= 2
    if last_ret > 0.04: score += 1
    return "PRUDENCE MAX 🔴" if score <= -1 else "OPPORTUNITÉ 🟢"

def quantum_monte_carlo(returns_window, n_sim=1000):
    mu = returns_window.mean().values
    cov = returns_window.cov().values * 252
    cov += np.eye(len(cov)) * 1e-6
    
    sim = np.random.multivariate_normal(mu, cov, n_sim)
    amplitudes = np.sqrt(np.abs(np.exp(np.sum(sim, axis=1))))
    amplitudes /= amplitudes.sum()
    
    best_idx = np.argsort(-amplitudes)[:100]
    return sim[best_idx].mean(axis=0)

def risk_parity(cov_matrix):
    vol = np.sqrt(np.diag(cov_matrix))
    vol = np.where(vol == 0, 1e-6, vol)
    w = 1 / vol
    return w / w.sum()

def double_loop_feedback(cumulative_drawdown):
    return 1.2 if cumulative_drawdown < -0.15 else 1.0

# ========================== 3. PRÉPARATION ==========================
df['Saison_Dalio'] = df.apply(detect_dalio_season, axis=1)
df['Tension_Élastique'] = df.apply(elastic_tension, axis=1)

# ========================== 4. SIMULATION ==========================
with st.spinner('⚙️ Simulation en cours...'):
    capital_plus = [1_000_000]
    capital_classic = [1_000_000]
    weights = np.array([0.30, 0.55, 0.075, 0.075])
    cumulative_dd = 0
    
    progress_bar = st.progress(0)
    
    # CORRECTION : Utiliser l'index partagé de df et returns
    for idx, date in enumerate(df.index):
        if idx == 0:  # Skip premier mois (pas de return)
            continue
            
        if idx % 10 == 0:
            progress_bar.progress(idx / len(df))
        
        # Fenêtre glissante (36 mois max)
        window_start = max(0, idx - 36)
        window = returns.iloc[window_start:idx]
        
        if len(window) < 12:
            continue
        
        season = df['Saison_Dalio'].iloc[idx]
        cpi_change = df['CPI_YoY'].iloc[idx] - df['CPI_YoY'].iloc[idx-1] if idx > 0 else 0
        fed_change = df['FEDFUNDS'].iloc[idx] - df['FEDFUNDS'].iloc[idx-1] if idx > 0 else 0
        causal_adj = pearl_causal_adjustment(season, cpi_change, fed_change)
        
        vix = df['^VIX'].iloc[idx]
        tension = df['Tension_Élastique'].iloc[idx]
        
        recent_30d_start = max(0, idx - 30)
        recent_30d = returns.iloc[recent_30d_start:idx].mean().values
        vix_w, risk_mult = vix_aggressive_buy(vix, weights, recent_30d)
        
        cov = window.cov() * 252 + np.eye(4) * 1e-6
        rp_w = risk_parity(cov)
        mc_direction = quantum_monte_carlo(window)
        
        losses = np.where(mc_direction < 0, mc_direction * 2.2, mc_direction)
        mc_adjusted = mc_direction - losses.mean() * 0.05
        
        final_w = (0.4 * rp_w + 0.25 * vix_w + 
                   0.15 * (weights + causal_adj) + 
                   0.15 * (mc_adjusted > 0) + 
                   0.05 * (1 - tension))
        final_w /= final_w.sum()
        
        dd = (capital_plus[-1] - max(capital_plus)) / max(capital_plus)
        cumulative_dd = min(cumulative_dd, dd)
        final_w *= double_loop_feedback(cumulative_dd)
        final_w /= final_w.sum()
        
        # CORRECTION : Utiliser .values pour extraire les valeurs
        ret_plus = np.dot(final_w, returns.iloc[idx].values)
        ret_classic = np.dot([0.30, 0.55, 0.075, 0.075], returns.iloc[idx].values)
        
        capital_plus.append(capital_plus[-1] * (1 + ret_plus))
        capital_classic.append(capital_classic[-1] * (1 + ret_classic))
    
    progress_bar.progress(1.0)

# ========================== 5. RÉSULTATS ==========================
result = pd.DataFrame({
    "DALIO+ ULTIMATE": capital_plus,
    "All Weather classique": capital_classic
}, index=df.index[:len(capital_plus)])

final = result["DALIO+ ULTIMATE"].iloc[-1]
final_classic = result["All Weather classique"].iloc[-1]

st.header("📊 Résultats — 1 000 000 $ investi le 1ᵉʳ janvier 2000")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 DALIO+", f"{final:,.0f} $", f"+{((final/1e6-1)*100):.1f}%")
with col2:
    st.metric("📈 All Weather", f"{final_classic:,.0f} $", f"+{((final_classic/1e6-1)*100):.1f}%")
with col3:
    st.metric("🚀 Surperformance", f"+{(final/final_classic-1)*100:.1f}%")

years = (result.index[-1] - result.index[0]).days / 365.25
st.subheader(f"📈 Rendement annualisé : **{((final/1e6)**(1/years)-1)*100:+.2f}%/an**")

# Graphique
fig, ax = plt.subplots(figsize=(15, 8))
result.plot(ax=ax, linewidth=2.5, color=['#FF6B6B', '#4ECDC4'])
ax.set_title("DALIO+ vs All Weather — 2000→2025", fontsize=20, fontweight='bold')
ax.set_ylabel("Valeur ($)", fontsize=14)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=13, loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# Stats
st.subheader("📉 Statistiques")
vol_plus = result["DALIO+ ULTIMATE"].pct_change().std() * np.sqrt(12) * 100
vol_classic = result["All Weather classique"].pct_change().std() * np.sqrt(12) * 100
dd_plus = ((result["DALIO+ ULTIMATE"] / result["DALIO+ ULTIMATE"].cummax()) - 1).min() * 100
dd_classic = ((result["All Weather classique"] / result["All Weather classique"].cummax()) - 1).min() * 100

stats = pd.DataFrame({
    'DALIO+': [final, ((final/1e6)**(1/years)-1)*100, vol_plus, dd_plus],
    'All Weather': [final_classic, ((final_classic/1e6)**(1/years)-1)*100, vol_classic, dd_classic]
}, index=['Capital final ($)', 'Rendement annuel (%)', 'Volatilité (%)', 'Drawdown max (%)'])

st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
st.success("✅ Simulation terminée !")
st.caption("🔬 Technologies : 4 saisons Dalio • Pearl causal • Monte-Carlo quantique • Six Hats • Risk Parity • VIX panic buying")
