# Databricks notebook source
# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
%matplotlib inline

# COMMAND ----------

# Python
df = pd.read_csv('./data/example_wp_peyton_manning.csv')
df.head()

# COMMAND ----------

# Python
def forecast(tdf):
  forecast_df = tdf[['ds', 'y']]
  
  m = Prophet()
  m.fit(forecast_df)
  
  future = m.make_future_dataframe(periods=365)
  forecast = m.predict(future)
  
  m.plot(forecast)
  
  return m, forecast

# COMMAND ----------

m, forecast = forecast(df)

# COMMAND ----------

# Python
m.plot_components(forecast);

