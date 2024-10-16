import streamlit as st
import numpy as np
import pandas as pd
import warnings
import riskfolio as rp
import matplotlib.pyplot as plt
from datetime import datetime
import locale

# Set locale to US for AM/PM format
locale.setlocale(locale.LC_TIME, "en_US")

# Streamlit page configuration
st.set_page_config(page_title="Danh mục HRP", layout="wide")
st.header("Danh mục đầu tư theo mô hình Hierarchical Risk Parity")
st.markdown(
    "Trong phần này, tôi sẽ tính toán danh mục đầu tư với rủi ro phân bổ ngang bằng giữa các tài sản trong danh mục (Gambeta và Kwon 2020) bằng cách sử dụng các phiên bản A, B và C của mô hình Relaxed Risk Parity và so sánh tỉ trọng giải ngân với mô hình của Markowitz 1952, lý thuyết danh mục đầu tư hiện đại. RRP là một mô hình cho phép kết hợp điều chỉnh tham số trong mô hình Vanilla Risk Parity."
)

# Create a file uploader to allow the user to upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Convert 'Date' column to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.set_index("Date", inplace=True)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Calculate returns
    Y = df.pct_change().dropna()

    # Building the portfolio object
    port = rp.Portfolio(returns=Y)

    # Calculating optimal portfolio
    method_mu = "hist"  # Method to estimate expected returns based on historical data
    method_cov = "hist"  # Method to estimate covariance matrix based on historical data
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Estimate optimal portfolio - Vanilla Risk Parity
    model = "Classic"
    rm = "MV"  # Risk measure used: variance
    rf = 0  # Risk-free rate
    b = None
    hist = True

    w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
    st.header("Tỉ trọng phân bổ theo mô hình Vanilla")
    st.dataframe(w_rp.T)

    # Plotting pie chart of allocations
    ax_pie = rp.plot_pie(
        w=w_rp,
        title="Risk Parity Variance",
        others=0.05,
        nrow=25,
        cmap="tab20",
        height=6,
        width=10,
    )
    st.pyplot(ax_pie.figure)

    # Plot risk contributions
    fig_risk_con, ax_risk_con = plt.subplots(figsize=(10, 6))
    ax_risk_con = rp.plot_risk_con(
        w_rp,
        cov=port.cov,
        returns=port.returns,
        rm=rm,
        rf=rf,
        alpha=0.01,
        color="tab:blue",
        height=6,
        width=10,
        ax=ax_risk_con,
    )
    st.pyplot(fig=fig_risk_con)

    # Plot portfolio table
    fig_table, ax_table = plt.subplots(figsize=(10, 6))
    ax_table = rp.plot_table(returns=Y, w=w_rp, MAR=0, alpha=0.05, ax=None)
    st.pyplot(fig=fig_table)

    # Plot histogram of returns
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist = rp.plot_hist(returns=Y, w=w_rp, alpha=0.05, bins=50, height=6, width=10)
    st.pyplot(fig=fig_hist)

    # Plot series returns
    fig_series, ax_series = plt.subplots(figsize=(10, 6))
    ax_series = rp.plot_series(returns=Y, w=w_rp, cmap="tab20", height=6, width=10)
    st.pyplot(fig=fig_series)

    # Model A - Relaxed Risk Parity Optimization
    version = "A"
    port.lowerret = 0.00056488 * 1.5
    w_rrp_a = port.rrp_optimization(model=model, version=version, l=1, b=b, hist=hist)
    st.header("Model A")
    st.dataframe(w_rrp_a.T)

    # Model B
    version = "B"
    w_rrp_b = port.rrp_optimization(model=model, version=version, l=1, b=b, hist=hist)
    st.header("Model B")
    st.dataframe(w_rrp_b.T)

    # Model C
    version = "C"
    w_rrp_c = port.rrp_optimization(model=model, version=version, l=1, b=b, hist=hist)
    st.header("Model C")
    st.dataframe(w_rrp_c.T)

    # Markowitz Optimization - MPT
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    w_mpt = port.optimization(model=model, rm=rm, obj="Sharpe", rf=rf, l=0, hist=hist)
    st.header("Tỉ trọng giải ngân theo mô hình lý thuyết MPT cổ điển")
    st.dataframe(w_mpt.T)

    # Plot efficient frontier
    ws = port.efficient_frontier(model="Classic", rm=rm, points=20, rf=0, hist=True)
    label = "Max Risk Adjusted Return Portfolio"
    fig_frontier, ax_frontier = plt.subplots(figsize=(10, 6))
    ax_frontier = rp.plot_frontier(
        w_frontier=ws,
        mu=port.mu,
        cov=port.cov,
        returns=port.returns,
        rm=rm,
        rf=0,
        alpha=0.05,
        cmap="viridis",
        w=w_mpt,
        label=label,
        marker="*",
        s=16,
        c="r",
        height=6,
        width=10,
        t_factor=252,
        ax=ax_frontier,
    )
    st.pyplot(fig=fig_frontier)

else:
    st.write("Please upload a CSV file to continue.")
