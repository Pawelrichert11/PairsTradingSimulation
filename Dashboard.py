import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import Config
from Simulation import PairTradingStrategy

st.set_page_config(layout="wide", page_title="Pairs Trading Dashboard")

def load_simulation_results():
    file_path = Config.PROCESSED_DIR / "simulation_results.parquet"
    
    if file_path.exists():
        df = pd.read_parquet(file_path)
        
        if 'coint_pvalue' not in df.columns:
            df['coint_pvalue'] = 1.0 
        
        cols_to_numeric = ['sharpe_ratio', 'total_return', 'coint_pvalue', 'annualized_return', 'correlation', 'final_value']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'annualized_return' not in df.columns:
            df['annualized_return'] = 0.0
            
        return df
    return pd.DataFrame()

def load_market_data():
    if Config.PROCESSED_MARKET_DATA.exists():
        return pd.read_parquet(Config.PROCESSED_MARKET_DATA)
    return pd.DataFrame()

def plot_pair_analysis(strategy):
    df = strategy.df.copy()
    
    df = df.dropna(subset=[strategy.ticker1, strategy.ticker2])
    
    window = strategy.window_size if hasattr(strategy, 'window_size') else 60
    
    rolling_corr = df[strategy.ticker1].rolling(window=window).corr(df[strategy.ticker2])
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            f"Price history & Strategy Performance", 
            "Z-Score & Trading Signals",
            f"Rolling {window}-Day Correlation"
        ),
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(go.Scatter(x=df.index, y=df[strategy.ticker1], name=strategy.ticker1, line=dict(color='blue', width=1), opacity=0.5), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df[strategy.ticker2], name=strategy.ticker2, line=dict(color='orange', width=1), opacity=0.5), row=1, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_return'], name='Equity', line=dict(color='purple', width=3)), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', line=dict(color='black', width=1)), row=2, col=1)
    fig.add_hline(y=strategy.std_dev_entry, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-strategy.std_dev_entry, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
    
    if 'signal' in df.columns:
        longs = df[df['signal'] == 1]
        shorts = df[df['signal'] == -1]
        fig.add_trace(go.Scatter(x=longs.index, y=longs['z_score'], mode='markers', marker=dict(color='green', size=6, symbol='triangle-up'), name='Long'), row=2, col=1)
        fig.add_trace(go.Scatter(x=shorts.index, y=shorts['z_score'], mode='markers', marker=dict(color='red', size=6, symbol='triangle-down'), name='Short'), row=2, col=1)

    # Row 3: Rolling Correlation
    fig.add_trace(go.Scatter(x=df.index, y=rolling_corr, name='Roll Corr', line=dict(color='teal', width=1.5), fill='tozeroy'), row=3, col=1)
    fig.add_hline(y=0, line_color="gray", line_dash="dot", row=3, col=1)
    
    # Konfiguracja Layoutu
    fig.update_layout(
        height=1000, 
        title_text=f"Analysis: {strategy.ticker1} vs {strategy.ticker2}", 
        dragmode='pan',
        yaxis=dict(title="Prices"), 
        yaxis2=dict(title="Equity", overlaying="y", side="right"),
        
        # --- USTAWIENIE ZAKRESU OSI X ---
        # Dzięki temu suwak na dole (rangeslider) dopasuje się do przyciętych danych
        xaxis3=dict(
            rangeslider=dict(visible=True), 
            type="date",
            range=[df.index[0], df.index[-1]] # Wymuszamy zakres na podstawie przyciętych danych
        )
    )
    return fig

# ---------------------------------------------------------
# HELPER: INTERAKTYWNY WYKRES SCATTER
# ---------------------------------------------------------
def render_interactive_chart(df, x_col, y_col, title, key_suffix, log_x=False):
    st.subheader(title)
    
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        st.warning(f"No data available for {title}")
        return

    # Slider do osi Y
    y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()
    buff = (abs(y_max) + abs(y_min)) * 0.1 if y_max != y_min else 0.1
    
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_range = st.slider(
            f"Filter Y-Axis ({y_col})", 
            min_value=float(y_min - buff), 
            max_value=float(y_max + buff), 
            value=(float(y_min - buff), float(y_max + buff)),
            key=f"slider_{key_suffix}"
        )
    else:
        y_range = None

    fig = px.scatter(
        plot_df, x=x_col, y=y_col, 
        trendline="ols", 
        hover_data=['pair', 'annualized_return', 'sharpe_ratio'], 
        log_x=log_x,
        color='sharpe_ratio', # Kolor kropek zależny od Sharpe
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_traces(line=dict(color='orange', width=2), selector=dict(mode='lines')) # Linia trendu
    fig.update_xaxes(rangeslider_visible=True)
    
    if y_range:
        fig.update_yaxes(range=y_range)
    
    if 'pvalue' in x_col or 'p_val' in x_col:
         fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="0.05")
         
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_suffix}")

def main():
    st.title("Pairs Trading Simulation Dashboard")
    st.sidebar.header("Configuration")
    
    # 1. Ładowanie danych
    with st.spinner("Loading simulation results..."):
        results_df = load_simulation_results()
        market_data = load_market_data()

    if results_df.empty:
        st.error("Brak pliku 'simulation_results.parquet'. Uruchom najpierw 'multisimulation.py'.")
        return

    # 3. Wybór głównej metryki
    profit_col = 'annualized_return' if 'annualized_return' in results_df.columns else 'total_return'
    st.sidebar.info(f"Primary Profit Metric: {profit_col.replace('_', ' ').title()}")
    
    # Wyświetlenie kosztu transakcji jeśli jest dostępny
    if 'transaction_cost_used' in results_df.columns:
        cost_used = results_df['transaction_cost_used'].iloc[0]
        st.sidebar.write(f"**Transaction Cost Used:** {cost_used:.2%}")

    # 4. Tabela wyników (Top Pairs)
    st.header("🏆 Top Performing Pairs")
    
    # Filtrowanie w sidebarze
    if 'sharpe_ratio' in results_df.columns:
        min_sharpe = st.sidebar.slider("Filter: Min Sharpe Ratio", -2.0, 5.0, 0.0, 0.1)
        filtered_results = results_df[results_df['sharpe_ratio'] >= min_sharpe].copy()
    else:
        filtered_results = results_df.copy()
    
    # Formatowanie tabeli
    fmt = {
        'total_return': '{:.2%}', 
        'annualized_return': '{:.2%}', 
        'sharpe_ratio': '{:.2f}', 
        'final_value': '{:.2f}', 
        'coint_pvalue': '{:.5f}', 
        'correlation': '{:.2f}'
    }
    cols_to_show = [c for c in ['pair', 'ticker_1', 'ticker_2', 'annualized_return', 'sharpe_ratio', 'coint_pvalue', 'correlation', 'total_return'] if c in filtered_results.columns]
    
    st.dataframe(
        filtered_results[cols_to_show].sort_values(by=profit_col, ascending=False).head(50).style.format({k: v for k, v in fmt.items() if k in cols_to_show})
    )

    st.markdown("---")

    # =========================================================================
    # 5. GLOBAL ANALYSIS (Scatter Plots)
    # =========================================================================
    st.header("📊 Global Analysis: Factors vs Annual Return")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Chart 1: Cointegration
        if 'coint_pvalue' in filtered_results.columns:
            render_interactive_chart(
                filtered_results, 'coint_pvalue', profit_col, 
                "Cointegration (P-Value) vs Return", "coint", log_x=False
            )
            
    with c2:
        # Chart 2: Correlation
        if 'correlation' in filtered_results.columns:
            render_interactive_chart(
                filtered_results, 'correlation', profit_col, 
                "Correlation vs Return", "corr"
            )

    # =========================================================================
    # 6. GROUPED ANALYSIS (BAR CHARTS) - COINTEGRATION & CORRELATION
    # =========================================================================
    st.markdown("---")
    st.header("📊 Grouped Analysis: Quality Buckets")

    col_grp1, col_grp2 = st.columns(2)

    # --- A. Wykres Kointegracji (istniejący) ---
    with col_grp1:
        if 'coint_pvalue' in filtered_results.columns:
            st.subheader("Performance by Cointegration")
            bins_coint = [-float('inf'), 0.01, 0.05, 0.10, float('inf')]
            labels_coint = ['Strong (< 0.01)', 'Significant (0.01-0.05)', 'Weak (0.05-0.10)', 'None (> 0.10)']
            
            df_cat = filtered_results.copy()
            df_cat['Coint_Quality'] = pd.cut(df_cat['coint_pvalue'], bins=bins_coint, labels=labels_coint)
            
            stats_coint = df_cat.groupby('Coint_Quality', observed=False).agg(
                avg_return=(profit_col, 'mean'), 
                count=(profit_col, 'count')
            ).reset_index()
            
            # Label
            stats_coint['label'] = stats_coint.apply(lambda x: f"{x['avg_return']:.2%} (n={int(x['count'])})", axis=1)
            
            fig_bar_coint = px.bar(
                stats_coint, x='Coint_Quality', y='avg_return', color='avg_return',
                color_continuous_scale='RdYlGn', text='label',
                labels={'avg_return': 'Avg Annual Return'}
            )
            fig_bar_coint.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar_coint, use_container_width=True)

    # --- B. Wykres Korelacji (NOWY) ---
    with col_grp2:
        if 'correlation' in filtered_results.columns:
            st.subheader("Performance by Correlation Strength")
            # Definiujemy koszyki korelacji
            bins_corr = [-1.0, 0.4, 0.7, 1.0]
            labels_corr = ['Low (< 0.4)', 'Medium (0.4-0.7)', 'High (> 0.7)']
            
            df_corr = filtered_results.copy()
            df_corr['Corr_Category'] = pd.cut(df_corr['correlation'], bins=bins_corr, labels=labels_corr, include_lowest=True)
            
            stats_corr = df_corr.groupby('Corr_Category', observed=False).agg(
                avg_return=(profit_col, 'mean'), 
                count=(profit_col, 'count')
            ).reset_index()
            
            # Usuwamy puste kategorie
            stats_corr = stats_corr.dropna(subset=['avg_return'])
            
            # Label
            stats_corr['label'] = stats_corr.apply(lambda x: f"{x['avg_return']:.2%} (n={int(x['count'])})", axis=1)
            
            fig_bar_corr = px.bar(
                stats_corr, x='Corr_Category', y='avg_return', color='avg_return',
                color_continuous_scale='RdYlGn', text='label',
                labels={'avg_return': 'Avg Annual Return', 'Corr_Category': 'Correlation Range'}
            )
            fig_bar_corr.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar_corr, use_container_width=True)

    st.markdown("---")

    if 'coint_pvalue' in filtered_results.columns and 'correlation' in filtered_results.columns:
            st.subheader("🎯 The Sweet Spot: Matrix Analysis (Cointegration + Correlation)")
            st.markdown("Average Annual Return for pairs falling into intersecting categories.")
            
            df_matrix = filtered_results.copy()
            
            # Kategoryzacja
            df_matrix['Coint_Quality'] = pd.cut(df_matrix['coint_pvalue'], bins=bins_coint, labels=labels_coint)
            df_matrix['Corr_Category'] = pd.cut(df_matrix['correlation'], bins=bins_corr, labels=labels_corr, include_lowest=True)
            
            # Grupowanie po obu wymiarach
            matrix_stats = df_matrix.groupby(['Coint_Quality', 'Corr_Category'], observed=False)[profit_col].mean().reset_index()
            
            # Tworzenie macierzy (pivot)
            heatmap_data = matrix_stats.pivot(index='Coint_Quality', columns='Corr_Category', values=profit_col)
            
            # Sortowanie logiczne wierszy i kolumn
            # Wiersze: Strong (najlepsze) na górze
            heatmap_data = heatmap_data.reindex(index=labels_coint)
            # Kolumny: Low -> High
            heatmap_data = heatmap_data.reindex(columns=labels_corr)

            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Correlation Strength", y="Cointegration Quality", color="Avg Return"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                text_auto='.2%', # Formatowanie tekstu w komórkach
                color_continuous_scale='RdYlGn', # Czerwony -> Zielony
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # 7. Return Distribution Analysis
    # =========================================================================
    st.header("Return Distribution Analysis")
    
    col_d1, col_d2 = st.columns(2)
    
    # 7a. Quantile Groups (Area Chart)
    with col_d1:
        st.subheader("Average Return per Quantile")
        st.markdown("Average return grouped by performance (Worst → Best).")
        df_dist = filtered_results.copy()
        
        if len(df_dist) >= 20:
            n_buckets = 20
            df_dist['quantile_rank'] = pd.qcut(df_dist[profit_col].rank(method='first'), q=n_buckets, labels=False)
            dist_stats = df_dist.groupby('quantile_rank')[profit_col].mean().reset_index()
            dist_stats['bucket_label'] = dist_stats['quantile_rank'].apply(lambda x: f"{x*5}-{(x+1)*5}%")
            
            fig_dist = px.area(
                dist_stats, 
                x='bucket_label', 
                y=profit_col, 
                markers=True,
                labels={profit_col: 'Avg Annual Return', 'bucket_label': 'Percentile Group'}
            )
            fig_dist.update_traces(line_color='#636EFA')
            fig_dist.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Not enough pairs for quantile buckets.")

    # 7b. NEW: NATURAL DISTRIBUTION (Probability Density)
    with col_d2:
        st.subheader("Natural Distribution (Density)")
        st.markdown("Shape of the return distribution (Frequency of results).")
        
        # Wersja kompatybilna ze starszym Plotly
        fig_dens = px.histogram(
            filtered_results, 
            x=profit_col, 
            nbins=50, 
            histnorm='probability density',
            labels={profit_col: 'Annual Return'},
            opacity=0.6
        )
        
        # Ręcznie dodajemy linię gęstości (KDE) zamiast 'element="poly"'
        # Używamy prostszego podejścia: histogram + marginal rug
        fig_dens.update_traces(marker_color='orange') 
        fig_dens.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Break Even")
        fig_dens.update_layout(hovermode="x unified", yaxis_title="Density")
        
        st.plotly_chart(fig_dens, use_container_width=True)

    st.markdown("---")
    # =========================================================================
    # 8. Single Pair Deep Dive
    # =========================================================================
    st.header("Single Pair Deep Dive")
    
    # Dropdown do wyboru pary
    if not filtered_results.empty:
        # Sortujemy pary alfabetycznie lub po wyniku, żeby łatwiej szukać
        pair_list = filtered_results.sort_values(by=profit_col, ascending=False)['pair'].tolist()
        sel_pair = st.selectbox("Select Pair to Analyze", pair_list)
        
        if sel_pair:
            # Pobierz dane dla wybranej pary
            row = filtered_results[filtered_results['pair'] == sel_pair].iloc[0]
            t1 = row['ticker_1']
            t2 = row['ticker_2']
            
            # Wyświetl metryki w kolumnach
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("CAGR", f"{row.get(profit_col, 0):.2%}")
            m2.metric("Sharpe", f"{row.get('sharpe_ratio', 0):.2f}")
            m3.metric("Correlation", f"{row.get('correlation', 0):.2f}")
            
            p_val = row.get('coint_pvalue', 1.0)
            m4.metric("Coint P-Value", f"{p_val:.4f}", delta="Strong" if p_val < 0.01 else "Weak", delta_color="inverse")
            m5.metric("Final Value", f"{row.get('final_value', 0):.2f}x")

            # Uruchomienie szczegółowej symulacji dla wykresu
            if not market_data.empty:
                try:
                    # Sprawdź czy mamy dane kosztów z pliku, jeśli nie, przyjmij domyślne 0.1%
                    cost = row.get('transaction_cost_used', 0.001)
                    
                    strategy = PairTradingStrategy(t1, t2, market_data)
                    strategy.run_backtest(transaction_cost=cost) # Używamy tego samego kosztu co w symulacji
                    
                    st.plotly_chart(plot_pair_analysis(strategy), use_container_width=True)
                    
                    with st.expander("Show Raw Trade Data"):
                        st.dataframe(strategy.df)
                        
                except Exception as e:
                    st.error(f"Error analyzing pair {sel_pair}: {e}")
            else:
                st.error("Market Data not loaded.")

if __name__ == "__main__":
    main()