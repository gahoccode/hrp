import streamlit as st
import numpy as np
import pandas as pd
import warnings
import riskfolio as rp
import investpy as ivp
import matplotlib.pyplot as plt
from IPython.display import display
from datetime import datetime

st.set_page_config(page_title = "Danh mục HRP", layout = "wide")
st.header("Danh mục đầu tư theo mô hình Hierarchical Risk Parity")
st.markdown('Trong phần này, tôi sẽ tính toán danh mục đầu tư với rủi ro phân bổ ngang bằng giữa các tài sản trong danh mục (Gambeta và Kwon 2020) bằng cách sử dụng các phiên bản A, B và C của mô hình Relaxed Risk Parity và so sánh tỉ trọng giải ngân với mô hình của Markowitz 1952, lý thuyết danh mục đầu tư hiện đại. RRP là một mô hình cho phép kết hợp điều chỉnh tham số trong mô hình Vanilla Risk Parity.')
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Ngày bắt đầu",datetime(2013, 1, 1))
with col2:
    end_date = st.date_input("Ngày kết thúc") # it defaults to current date
try: 
    tickers_string = st.text_input('Nhập mã chứng khoán cách nhau bởi dấu phẩy không khoảng trắng, vd: "REE,PHR,FMC"', 'REE,PHR,FMC').upper()
    tickers = tickers_string.split(',')
except:
    st.write('Enter correct stock tickers to be included in portfolio separated by commas WITHOUT spaces, e.g. "REE,PHR,FMC"and hit Enter.')

country_of_choice = "vietnam"
data_frequency = "Daily"

# initializing the final dataframe
stocks_df = pd.DataFrame()

# looping through each entries in our stock list to get data
for i in range(len(tickers)):
    
    # making calls using investpy api to get historical data
    current_stock_df = ivp.stocks.get_stock_historical_data(stock = tickers[i], country = country_of_choice, from_date = start_date.strftime('%d/%m/%Y'), to_date = end_date.strftime('%d/%m/%Y'), as_json = False, order = 'ascending', interval = data_frequency)
    
    # setting data with close prices
    stocks_df[tickers[i]] = current_stock_df["Close"]
    
# dropping the rows if any of the columns has nan value
data = stocks_df.dropna()
Y = data.pct_change().dropna()
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic' # Could be Classic (historical) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
b = None # Risk contribution constraints vector

w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
st.header('Tỉ trọng phân bổ theo mô hình Vanilla')
st.dataframe(w_rp.T)

ax = rp.plot_pie(w=w_rp, title='Risk Parity Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)

fig= ax.figure
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
ax1 = rp.plot_risk_con(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=ax)

fig1= ax1.figure
st.pyplot(fig=fig)

st.header('Model A')
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic' # Could be Classic (historical) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
b = None # Risk contribution constraints vector
version = 'A' # Could be A, B or C
l = 1 # Penalty term, only valid for C version

# Setting the return constraint
port.lowerret = 0.00056488 * 1.5

w_rrp_a = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)

st.dataframe(w_rrp_a.T)

st.header('Model B')

# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic' # Could be Classic (historical) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
version = 'B' # Could be A, B or C

w_rrp_b = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)

st.dataframe(w_rrp_b.T)

st.header('Model C')

version = 'C' # Could be A, B or C

w_rrp_c = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)

st.dataframe(w_rrp_c.T)



# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
st.header('Tỉ trọng giải ngân theo mô hình lý thuyết MPT cổ điển')
st.dataframe(w.T)

