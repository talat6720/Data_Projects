import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Assuming 'df1' is the DataFrame that contains the time series data
# and that 'DateTime' is the column with time stamps and 'Consumption' is the target variable.
df1 = pd.read_csv('electricityConsumptionAndProductioction.csv')

# First, we need to make sure that 'DateTime' is the index and in datetime format
df1['DateTime'] = pd.to_datetime(df1['DateTime'])
df1.set_index('DateTime', inplace=True)

# Perform seasonal decomposition
# The model can be 'additive' or 'multiplicative', depending on the nature of the series.
# An additive model is linear where changes over time are consistently made by the same amount.
# A multiplicative model is nonlinear, such as when changes increase or decrease over time.

# We will use an additive model for this example.
decomposition = seasonal_decompose(df1['Consumption'], model='additive', period=24)  # Assuming hourly data with daily seasonality

# Plot the decomposed components
decomposed_components = decomposition.plot()
plt.show()
