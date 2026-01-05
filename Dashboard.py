import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import Config
from Simulation import PairTradingStrategy

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Pairs Trading Dashboard")

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_simulation_results():
    file_path = Config.PROCESSED_DIR / "simulation_results.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'coint_pvalue' not in df.columns:
            df['coint_pvalue'] = 1.0 
        
        # Force numeric types
        for col in ['sharpe_ratio', 'total_return', 'coint_pvalue', 'annualized_return']:
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

@st.cache_data
def ensure_correlation_data(results_df, market_data):
    if 'correlation' not in results_df.columns:
        corrs = []
        for i, row in results_df.iterrows():
            try:
                if 'ticker_1' in row and 'ticker_2' in row:
                    t1, t2 = row['ticker_1'], row['ticker_2']
                else:
                    parts = row['pair'].split('-')
                    t1, t2 = parts[0], parts[1]
                
                if t1 in market_data.columns and t2 in market_data.columns:
                    c = market_data[t1].corr(market_data[t2])
                else:
                    c = 0.0
                corrs.append(c)
            except:
                corrs.append(0.0)
        results_df['correlation'] = corrs
    return results_df

# ---------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_pair_analysis(strategy):
    df = strategy.df
    perf_col = 'cum_return' if 'cum_return' in df.columns else None
    if not perf_col and 'strategy_return' in df.columns:
        df['cum_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        perf_col = 'cum_return'
    
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

    # Row 1
    fig.add_trace(go.Scatter(x=df.index, y=df[strategy.ticker1], name=strategy.ticker1, line=dict(color='blue', width=1), opacity=0.5), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df[strategy.ticker2], name=strategy.ticker2, line=dict(color='orange', width=1), opacity=0.5), row=1, col=1, secondary_y=False)
    if perf_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[perf_col], name='Equity', line=dict(color='purple', width=3)), row=1, col=1, secondary_y=True)

    # Row 2
    fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', line=dict(color='black', width=1)), row=2, col=1)
    fig.add_hline(y=strategy.std_dev_entry, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-strategy.std_dev_entry, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
    
    longs = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    fig.add_trace(go.Scatter(x=longs.index, y=longs['z_score'], mode='markers', marker=dict(color='green', size=6, symbol='triangle-up'), name='Long'), row=2, col=1)
    fig.add_trace(go.Scatter(x=shorts.index, y=shorts['z_score'], mode='markers', marker=dict(color='red', size=6, symbol='triangle-down'), name='Short'), row=2, col=1)

    # Row 3
    fig.add_trace(go.Scatter(x=df.index, y=rolling_corr, name='Roll Corr', line=dict(color='teal', width=1.5), fill='tozeroy'), row=3, col=1)
    fig.add_hline(y=0, line_color="gray", line_dash="dot", row=3, col=1)
    
    fig.update_layout(height=1000, title_text="Detailed Strategy Analysis", dragmode='pan',
                      yaxis=dict(title="Prices"), yaxis2=dict(title="Equity", overlaying="y", side="right"),
                      xaxis3=dict(rangeslider=dict(visible=True), type="date"))
    return fig

# ---------------------------------------------------------
# HELPER: RENDER INTERACTIVE CHART WITH SLIDERS
# ---------------------------------------------------------
def render_interactive_chart(df, x_col, y_col, title, key_suffix, log_x=False):
    """
    Renders a single scatter plot with:
    1. Streamlit Slider for Y-Axis Zoom
    2. Plotly Range Slider for X-Axis Zoom
    3. Orange Trendline
    """
    st.subheader(title)
    
    # 1. Prepare Data
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if plot_df.empty:
        st.warning(f"No data available for {title}")
        return

    # 2. Create Y-Axis Slider (Streamlit)
    y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()
    buff = (abs(y_max) + abs(y_min)) * 0.1 if y_max != y_min else 0.1
    
    # Check for NaN/Inf
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_range = st.slider(
            f"Filter Y-Axis (Return)", 
            min_value=float(y_min - buff), 
            max_value=float(y_max + buff), 
            value=(float(y_min - buff), float(y_max + buff)),
            key=f"slider_{key_suffix}"  # Unique key for slider
        )
    else:
        y_range = None

    # 3. Create Scatter Plot
    fig = px.scatter(
        plot_df, x=x_col, y=y_col, 
        trendline="ols", 
        hover_data=['pair'], 
        log_x=log_x
    )
    
    # 4. Orange Trendline
    fig.update_traces(line=dict(color='orange'), selector=dict(mode='lines'))

    # 5. Enable X-Axis Range Slider (Plotly)
    fig.update_xaxes(rangeslider_visible=True)
    
    # 6. Apply Y-Axis Zoom
    if y_range:
        fig.update_yaxes(range=y_range)
    
    # 7. Add Significance Line for P-Value
    if 'pvalue' in x_col or 'p_val' in x_col:
         fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="0.05")
         
    # FIX: Add unique key to plotly_chart to prevent StreamlitDuplicateElementId
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_suffix}")


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
def main():
    st.title("Pairs Trading Simulation Dashboard")
    st.sidebar.header("Configuration")
    
    results_df = load_simulation_results()
    market_data = load_market_data()

    if results_df.empty or market_data.empty:
        st.error("Missing data files. Run 'LoadData.py' and 'MultiSimulationOneProcess.py'.")
        return

    results_df = ensure_correlation_data(results_df, market_data)

    profit_col = 'annualized_return' if 'annualized_return' in results_df.columns else 'total_return'
    st.sidebar.info(f"Primary Profit Metric: {profit_col.replace('_', ' ').title()}")

    # 1. Top Performing Pairs Table
    st.header("🏆 Top Performing Pairs")
    if 'sharpe_ratio' in results_df.columns:
        min_sharpe = st.sidebar.slider("Filter: Min Sharpe Ratio", -2.0, 5.0, 0.0, 0.1)
        filtered_results = results_df[results_df['sharpe_ratio'] >= min_sharpe].copy()
    else:
        filtered_results = results_df.copy()
    
    fmt = {
        'total_return': '{:.2%}', 
        'annualized_return': '{:.2%}', 
        'sharpe_ratio': '{:.2f}', 
        'final_value': '{:.2f}', 
        'coint_pvalue': '{:.5f}', 
        'correlation': '{:.2f}'
    }
    st.dataframe(filtered_results.style.format({k: v for k, v in fmt.items() if k in filtered_results.columns}))

    st.markdown("---")

    # =========================================================================
    # 2. GLOBAL ANALYSIS (3 Stacked Charts with Sliders)
    # =========================================================================
    st.header("📊 Global Analysis: Factors vs Annual Return")
    
    target_y = 'annualized_return' if 'annualized_return' in filtered_results.columns else profit_col

    # Chart 1: Cointegration (FIXED: log_x=False for linear scale)
    if 'coint_pvalue' in filtered_results.columns:
        render_interactive_chart(
            filtered_results, 'coint_pvalue', target_y, 
            "1. Cointegration vs Return (Linear Scale)", "coint", log_x=False
        )

    # Chart 2: Correlation
    if 'correlation' in filtered_results.columns:
        render_interactive_chart(
            filtered_results, 'correlation', target_y, 
            "2. Correlation vs Return", "corr"
        )

    # Chart 3: Sharpe
    if 'sharpe_ratio' in filtered_results.columns:
        render_interactive_chart(
            filtered_results, 'sharpe_ratio', target_y, 
            "3. Sharpe Ratio vs Return", "sharpe"
        )

    st.markdown("---")

    # 3. DETAILED STRATEGY VIEW
    st.header("Detailed Strategy View")
    if 'pair' in filtered_results.columns and not filtered_results.empty:
        pair_list = filtered_results['pair'].tolist()
        sel_pair = st.selectbox("Select Pair", pair_list) if pair_list else None
        
        if sel_pair:
            row = filtered_results[filtered_results['pair'] == sel_pair].iloc[0]
            t1 = row.get('ticker_1', sel_pair.split('-')[0])
            t2 = row.get('ticker_2', sel_pair.split('-')[1] if '-' in sel_pair else '')
            
            c1, c2, c3, c4 = st.columns(4)
            
            ann_ret_val = row.get('annualized_return', 0)
            c1.metric("Annual Return (CAGR)", f"{ann_ret_val:.2%}")
            c2.metric("Sharpe", f"{row.get('sharpe_ratio', 0):.2f}")
            
            if 'coint_pvalue' in row: 
                p_val = row['coint_pvalue']
                c3.metric("Coint P-Value", f"{p_val:.4f}", delta="Significant" if p_val < 0.05 else "Weak", delta_color="inverse")
            
            c4.metric("Total Return", f"{row.get('total_return', 0):.2%}")

            try:
                strategy = PairTradingStrategy(t1, t2, market_data)
                strategy.run_backtest()
                st.plotly_chart(plot_pair_analysis(strategy), use_container_width=True)
                
                with st.expander("View Raw Data"):
                    st.dataframe(strategy.df.tail(100))
            except Exception as e:
                st.error(f"Error running detailed simulation: {e}")

    # =========================================================================
    # 4. AVERAGE ANNUAL PERFORMANCE BY COINTEGRATION
    # =========================================================================
    if 'coint_pvalue' in filtered_results.columns and profit_col in filtered_results.columns and not filtered_results.empty:
        st.subheader(f"📊 Average Annual Performance by Cointegration Quality")
        
        bins = [-float('inf'), 0.01, 0.05, 0.10, float('inf')]
        labels = ['Strong (< 0.01)', 'Significant (0.01-0.05)', 'Weak (0.05-0.10)', 'None (> 0.10)']
        
        df_cat = filtered_results.copy()
        df_cat['Coint_Quality'] = pd.cut(df_cat['coint_pvalue'], bins=bins, labels=labels)
        
        stats = df_cat.groupby('Coint_Quality', observed=True).agg(
            avg_return=('annualized_return', 'mean'), 
            count=('annualized_return', 'count')
        ).reset_index()
        
        if len(filtered_results) > 0:
            stats['share'] = stats['count'] / len(filtered_results)
        else:
            stats['share'] = 0.0

        stats['label'] = stats.apply(lambda x: f"{x['avg_return']:.2%} (n={int(x['count'])})", axis=1)
        
        bar_y_min, bar_y_max = float(stats['avg_return'].min()), float(stats['avg_return'].max())
        buff = (abs(bar_y_max) + abs(bar_y_min)) * 0.1 if bar_y_max != bar_y_min else 0.1
        bar_rng = st.slider("Bar Chart Zoom", bar_y_min-buff, bar_y_max+buff, (bar_y_min-buff, bar_y_max+buff), label_visibility="collapsed")

        fig_bar = px.bar(
            stats, x='Coint_Quality', y='avg_return', color='avg_return', 
            title=f"Average Annualized Return per Cointegration Level",
            labels={'avg_return': 'Avg Annualized Return', 'Coint_Quality': 'Cointegration Strength'}, 
            color_continuous_scale='RdYlGn', text='label'
        )
        fig_bar.update_xaxes(title="Cointegration P-Value Range", rangeslider_visible=False)
        fig_bar.update_yaxes(range=bar_rng, title="Avg Annual Return")
        fig_bar.update_traces(textposition='auto')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 5. HISTOGRAM
    st.subheader(f"📊 Distribution of {profit_col.replace('_', ' ').title()} (Rozkład Zysków)")
    if profit_col in filtered_results.columns and not filtered_results.empty:
        mean_ret = filtered_results[profit_col].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_ret:.2%}")
        c2.metric("Median", f"{filtered_results[profit_col].median():.2%}")
        c3.metric("Max", f"{filtered_results[profit_col].max():.2%}")
        c4.metric("Min", f"{filtered_results[profit_col].min():.2%}", delta_color="inverse")

        fig_hist = px.histogram(filtered_results, x=profit_col, nbins=100, title=f"Histogram of {profit_col}", color_discrete_sequence=['#636EFA'])
        fig_hist.update_xaxes(rangeslider_visible=True, tickformat='.0%')
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
        fig_hist.add_vline(x=mean_ret, line_dash="dot", line_color="yellow")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # 6. PERFORMANCE CURVE
    st.subheader("📈 Performance Distribution Profile")
    y_opts = [c for c in ['annualized_return', 'total_return', 'final_value', 'sharpe_ratio', 'coint_pvalue'] if c in filtered_results.columns]
    
    if y_opts and not filtered_results.empty:
        default_curve_idx = 0
        if profit_col in y_opts:
            default_curve_idx = y_opts.index(profit_col)
            
        y_col = st.selectbox("Select Metric (Y-Axis)", y_opts, index=default_curve_idx)

        if y_col:
            df_sorted = filtered_results.sort_values(by=y_col).reset_index(drop=True)
            df_sorted['percentile'] = (df_sorted.index / (len(df_sorted) - 1))
            
            y_min_l, y_max_l = float(df_sorted[y_col].min()), float(df_sorted[y_col].max())
            buff_l = (abs(y_max_l) + abs(y_min_l)) * 0.05
            line_rng = st.slider("Line Chart Zoom", y_min_l-buff_l, y_max_l+buff_l, (y_min_l-buff_l, y_max_l+buff_l), label_visibility="collapsed")

            fig_line = px.line(df_sorted, x="percentile", y=y_col, title=f"Curve: {y_col} by Percentile",
                               labels={'percentile': 'Percentile (Worst -> Best)'})
            fig_line.update_traces(line=dict(color='cyan', width=2))
            fig_line.update_xaxes(rangeslider_visible=True, tickformat='.0%')
            fig_line.update_yaxes(range=line_rng)
            st.plotly_chart(fig_line, use_container_width=True)

if __name__ == "__main__":
    main()