import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit

# Loa the data into a DataFrame 'df'
df = pd.read_csv('electricityConsumptionAndProductioction.csv')

# Sample data creation for demonstration (Replace this with above line to use your data)
np.random.seed(42)
date_rng = pd.date_range(start='2019-01-01', end='2019-03-01', freq='H')
df = pd.DataFrame(date_rng, columns=['DateTime'])
df['Consumption'] = np.random.uniform(low=4000, high=9000, size=(len(date_rng)))
df.set_index('DateTime', inplace=True)

# Define the SARIMA parameters
p = 2
d = 0
q = 2
P = 1
D = 1
Q = 1
s = 24  # Assuming hourly data and daily seasonality

# Prepare the data for time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)  # Number of splits for cross-validation

# Lists to store results of cross-validation
rmse_scores = []
models = []

# Perform time series cross-validation
for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Fit the SARIMA model
    model = SARIMAX(train['Consumption'],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    # Fit the model
    model_fit = model.fit(disp=False)
    
    # Store the fitted models
    models.append(model_fit)

    # Make predictions
    predictions = model_fit.forecast(steps=len(test))

    # Calculate and store RMSE
    score = rmse(test['Consumption'], predictions)
    rmse_scores.append(score)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 4))
    plt.plot(train['Consumption'], label='Train')
    plt.plot(test['Consumption'], label='Test')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(model_fit.resid, ax=axes[0])
    plot_pacf(model_fit.resid, ax=axes[1])
    plt.show()

# Print the RMSE scores
print(f"RMSE scores for each fold: {rmse_scores}")
print(f"Average RMSE score: {np.mean(rmse_scores)}")

# The final model will be chosen based on the lowest RMSE score
final_model = models[np.argmin(rmse_scores)]

# You can use final_model to make predictions on new data.
