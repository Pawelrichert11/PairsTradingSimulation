import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Importy lokalne (Twoje moduły)
from DatabaseManager import DatabaseManager
from Simulation import run_full_simulation

# --- KONFIGURACJA ŚCIEŻEK I STRONY ---
st.set_page_config(page_title="Interactive Algo Dashboard", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
PARQUET_FILE = BASE_DIR / "processed_files" / "processed_market_data.parquet"

# --- FUNKCJE POMOCNICZE ---

@st.cache_data
def load_data():
    """Ładuje wyniki z SQL i ceny z Parquet."""
    db = DatabaseManager()
    
    # 1. Wyniki z SQL
    try:
        results_df = db.get_all_simulation_results_sorted()
    except Exception as e:
        st.error(f"Błąd SQL: {e}. Uruchom najpierw symulację!")
        st.stop()

    # 2. Ceny z Parquet
    if not PARQUET_FILE.exists():
        st.error("Brak pliku Parquet z cenami. Uruchom LoadData.py.")
        st.stop()
    prices_df = pd.read_parquet(PARQUET_FILE)
    
    return results_df, prices_df

def plot_interactive_pair(prices_df, t1, t2, params):
    """
    Tworzy interaktywne wykresy Plotly dla danej pary.
    """
    # Pobranie danych dla pary
    pair_data = prices_df[[t1, t2]].dropna()
    
    # Uruchomienie symulacji "w locie" dla wykresów
    sim_res = run_full_simulation(pair_data, t1, t2, *params)
    
    if sim_res is None:
        return None, None

    # --- WYKRES 1: Equity Curve (Interactive) ---
    equity_pct = (sim_res['Cumulative_Return_Net'] - 1) * 100
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_pct.index, 
        y=equity_pct, 
        mode='lines', 
        name='Strategia',
        line=dict(color='#00CC96', width=2)
    ))
    fig_equity.add_hline(y=0, line_dash="dash", line_color="white")
    
    fig_equity.update_layout(
        title=f"Wynik Narastająco: {t1} vs {t2}",
        xaxis_title="Data",
        yaxis_title="Zysk (%)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark"
    )

    # --- WYKRES 2: Price Overlay (Interactive Zoom) ---
    # Normalizacja
    norm_t1 = pair_data[t1] / pair_data[t1].iloc[0]
    norm_t2 = pair_data[t2] / pair_data[t2].iloc[0]
    
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(x=norm_t1.index, y=norm_t1, name=t1, line=dict(color='#636EFA')))
    fig_prices.add_trace(go.Scatter(x=norm_t2.index, y=norm_t2, name=t2, line=dict(color='#EF553B')))
    
    # Dodajemy cieniowanie, gdy pozycja jest aktywna (opcjonalne, dla czytelności)
    # Dla uproszczenia w interaktywnym wykresie zostawiamy czyste linie, 
    # użytkownik może sam przybliżyć (zoom) momenty rozjazdu.

    fig_prices.update_layout(
        title=f"Porównanie Cen (Start = 1.0)",
        xaxis_title="Data",
        yaxis_title="Znormalizowana Cena",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        hovermode="x unified" # Wspólny tooltip dla obu linii
    )

    return fig_equity, fig_prices

# --- GŁÓWNA APLIKACJA ---

st.title("🚀 Interaktywny Dashboard Algo-Trading")
st.markdown("Eksploracja strategii Pairs Trading z wykorzystaniem **SQL, Parquet i Plotly**.")

# Ładowanie danych
results_df, all_prices_df = load_data()

if len(results_df) < 5:
    st.warning("Za mało danych. Uruchom symulację na większej liczbie par.")
    st.stop()

# Podział na grupy (Slicing)
N = 5
top_5 = results_df.head(N)
mid_start = len(results_df) // 2
mid_5 = results_df.iloc[mid_start : mid_start + N]
bottom_5 = results_df.tail(N)

# Dodajemy etykiety grup do DataFrame (dla wykresów grupowych)
results_df['Group'] = 'Others'
results_df.loc[top_5.index, 'Group'] = 'Top'
results_df.loc[bottom_5.index, 'Group'] = 'Bottom'
results_df.loc[mid_5.index, 'Group'] = 'Middle'

# --- SEKCJA 1: GLOBALNA ANALIZA (Plotly Express) ---
st.header("1. Analiza Globalna (Interactive)")

col1, col2 = st.columns(2)

with col1:
    # 1. SCATTER PLOT (Korelacja vs Wynik)
    st.subheader("Mapa Rynku (Scatter)")
    st.caption("Najedź na punkty, aby zobaczyć, które to pary!")
    
    fig_scatter = px.scatter(
        results_df, 
        x="korelacja", 
        y="wynik_netto", 
        color="wynik_netto",
        color_continuous_scale="RdYlGn",
        hover_data=["ticker_a", "ticker_b", "liczba_transakcji"], # TO JEST KLUCZOWE!
        title="Korelacja vs Wynik Netto"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # 2. HISTOGRAM (Rozkład wyników)
    st.subheader("Dystrybucja Zysków")
    fig_hist = px.histogram(
        results_df, 
        x="wynik_netto", 
        nbins=40, 
        title="Rozkład wyników wszystkich par",
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_hist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # 3. BOX PLOT (Statystyka wg Grup) - NOWOŚĆ
    st.subheader("Statystyka wg Grup (Box Plot)")
    # Filtrujemy tylko oznaczone grupy, żeby wykres był czytelny
    subset = results_df[results_df['Group'].isin(['Top', 'Middle', 'Bottom'])]
    
    fig_box = px.box(
        subset, 
        x="Group", 
        y="wynik_netto", 
        color="Group",
        points="all", # Pokazuje też pojedyncze punkty
        title="Rozrzut wyników w grupach"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col4:
    # 4. HEATMAP (Korelacja próbki) - NOWOŚĆ
    st.subheader("Macierz Korelacji (Top 5 Par)")
    # Bierzemy tickery z Top 5 i tworzymy macierz
    top_tickers = list(set(top_5['ticker_a'].tolist() + top_5['ticker_b'].tolist()))
    if len(top_tickers) > 1:
        corr_matrix = all_prices_df[top_tickers].corr()
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Korelacje między składnikami Top 5"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# --- SEKCJA 2: SZCZEGÓŁOWA INSPEKCJA (Plotly Graph Objects) ---
st.header("2. Szczegółowy Przegląd (Deep Dive)")
st.markdown("Przybliżaj wykresy (zoom), aby analizować momenty wejścia i wyjścia.")

# Parametry symulacji
PARAMS = (20, 2.0, 0.5, 0.001)

tab_top, tab_mid, tab_bot = st.tabs(["🏆 Top 5", "⚖️ Middle 5", "📉 Bottom 5"])

def render_tab_content(pairs_df):
    for i, row in pairs_df.iterrows():
        t1, t2 = row['ticker_a'], row['ticker_b']
        
        with st.expander(f"{t1} / {t2} | Wynik: {row['wynik_netto']*100:.2f}%", expanded=(i == pairs_df.index[0])):
            # Metryki w rzędzie
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Zysk Netto", f"{row['wynik_netto']*100:.2f}%")
            c2.metric("Transakcje", row['liczba_transakcji'])
            c3.metric("Korelacja", f"{row['korelacja']:.2f}")
            c4.metric("Avg Trade", f"{(row['wynik_netto']/max(1, row['liczba_transakcji'])*100):.2f}%")
            
            # Wykresy
            f_eq, f_pr = plot_interactive_pair(all_prices_df, t1, t2, PARAMS)
            
            if f_eq:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.plotly_chart(f_eq, use_container_width=True)
                with col_chart2:
                    st.plotly_chart(f_pr, use_container_width=True)

with tab_top:
    st.info("Pary o najwyższym zysku netto. Zwróć uwagę na stabilny wzrost equity.")
    render_tab_content(top_5)

with tab_mid:
    st.info("Pary przeciętne. Często duża liczba transakcji zjada zysk (prowizje).")
    render_tab_content(mid_5)

with tab_bot:
    st.error("Pary stratne. Zobacz na prawym wykresie jak ceny rozjeżdżają się permanentnie (brak konwergencji).")
    render_tab_content(bottom_5)