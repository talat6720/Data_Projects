import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools

# Load the dataset
# Loa the data into a DataFrame 'df'
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')

        
# Function to calculate RMSE
def evaluate_predictions(pred, true):
    mse = mean_squared_error(true, pred)
    rmse = sqrt(mse)
    print(f'RMSE: {rmse}')
    return rmse

# Assuming 'Consumption' is the main series of interest
check_stationarity(df['Consumption'])


# Define the TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)

# This will print the indices of the train and test sets for each split
for train_index, test_index in tscv.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)
    train, test = df.iloc[train_index], df.iloc[test_index]


# Placeholder lists for storing evaluation metrics for each split
metrics = []

for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Define and fit the model on the training set
    model = SARIMAX(train['Consumption'], order=(2,0,2), seasonal_order=(1,1,1,24))
    model_fit = model.fit(disp=False)

    # Forecast
    predictions = model_fit.forecast(steps=len(test))
    
    # Evaluate predictions
    # Here, you need to define a function to evaluate your predictions, e.g., RMSE
    metric = evaluate_predictions(predictions, test['Consumption'])
    metrics.append(metric)

# Calculate and print the average metric
average_metric = sum(metrics) / len(metrics)
print(f'Average Metric across folds: {average_metric}')


# Visualize the fit on the last fold
plt.figure(figsize=(10,5))
plt.plot(test.index, test['Consumption'], label='Observed')
plt.plot(test.index, predictions, label='Predicted')
plt.legend()
plt.show()

# Perform diagnostics on the last model fit
model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()


# Define the p, d, and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, d, q, and s quartets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

# Grid search
best_score, best_cfg = float("inf"), None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train['Consumption'], 
                            order=param, 
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            model_fit = model.fit(disp=False)
            
            # Use AIC as the evaluation metric
            aic = model_fit.aic
            if aic < best_score:
                best_score, best_cfg = aic, (param, param_seasonal)
            print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, aic))
        except:
            continue

print('Best SARIMA{}x{} - AIC:{}'.format(best_cfg[0], best_cfg[1], best_score))



# Fit the model with the best parameters found
best_model = SARIMAX(df['Consumption'], 
                     order=best_cfg[0], 
                     seasonal_order=best_cfg[1],
                     enforce_stationarity=False,
                     enforce_invertibility=False)

best_model_fit = best_model.fit(disp=False)

