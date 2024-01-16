import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

# Assuming the CSV file is located in the same directory as your Python script
df = pd.read_csv('electricityConsumptionAndProductioction.csv')

# Print the DataFrame
print(df)

print(df.columns)

# Calculate the minimum and maximum dates
min_date = df['DateTime'].min()
max_date = df['DateTime'].max()

# Print the range
print(f"Date range: {min_date} to {max_date}")

df['DateTime'] = pd.to_datetime(df['DateTime'])

# Create DataFrames for each year
df_2019 = df[df['DateTime'].dt.year == 2019]
df_2020 = df[df['DateTime'].dt.year == 2020]
df_2021 = df[df['DateTime'].dt.year == 2021]

# Create a DataFrame for the rest of the years
df_rest = df[~df['DateTime'].dt.year.isin([2019, 2020, 2021])]

# Print the first few rows to verify
print("2019 DataFrame:")
print(df_2019.head())
print("\n2020 DataFrame:")
print(df_2020.head())
print("\n2021 DataFrame:")
print(df_2021.head())
print("\nRest of the years DataFrame:")
print(df_rest.tail())

df.info()

# set the DateTime as the index
df = df.set_index("DateTime")
df.index = pd.to_datetime(df.index)

df.drop(['Production', 'Nuclear', 'Wind','Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass'], axis=1, inplace=True)

df[["Consumption"]].plot(style="-", figsize=(15, 5), title="Electricity Consumption, in MW")
plt.ylabel('MW')
# plt.show()

df["2023-01-09 00:00:00" : "2023-01-22 23:59:59"][["Consumption"]].plot(style="-", figsize=(15, 5), title="Electricity Consumption, in MW")
plt.ylabel('MW')
# plt.show()

# method for adding time features by the time index
def createTimeFeatures(df):
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day_of_year"] = df.index.dayofyear


# apply the method to the existing dataframe
createTimeFeatures(df)

import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday

# Define standard Romanian holidays
class RomanianHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year\'s Day', month=1, day=1),
        Holiday('Day after New Year\'s Day', month=1, day=2),
        Holiday('Labor Day', month=5, day=1),
        Holiday('Children\'s Day', month=6, day=1),
        Holiday('St. Mary\'s Day', month=8, day=15),
        Holiday('St. Andrew\'s Day', month=11, day=30),
        Holiday('National Day', month=12, day=1),
        Holiday('Christmas Day', month=12, day=25),
        Holiday('Second Day of Christmas', month=12, day=26)
    ]

# Create an instance of the holiday calendar and get the holidays
cal = RomanianHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())

# Add a 'holiday' column to the DataFrame
df['holiday'] = df.index.normalize().isin(holidays)


# Add a 'is_weekend' column to the DataFrame
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Show the first few rows of the updated DataFrame
df.head()


cutOffDate = df.index[-365 * 24]
print(f"cutOffDate {cutOffDate}")

train = df.loc[df.index <= cutOffDate]
test = df.loc[df.index > cutOffDate]

print(f"train size: {len(train)} and test {len(test)}")

df[["Consumption"]].plot(style="-", figsize=(15, 5), title="Electricity Consumption, in MW")
plt.ylabel('MW')
plt.axvline(x=cutOffDate, color='r')
# plt.show()

def meanAbsolutErrorAaPercentage(real, predicted):
    real = np.array(real)
    predicted = np.array(predicted)

    return np.mean(np.abs((real - predicted) / real)) * 100

FEATURES = ['hour', 'day_of_week', 'quarter', 'month', 'year','day_of_year', 'holiday', 'is_weekend']
# FEATURES = ['hour', 'day_of_week', 'quarter', 'month', 'year','day_of_year', 'is_weekend']
TARGET = "Consumption"

prophetTrain = train.reset_index()
prophetTrain.drop(FEATURES, axis=1, inplace=True)
prophetTrain.rename(columns={"DateTime": "ds", "Consumption": "y"}, inplace=True)

prophetTrain.tail()


prophetTest = test.reset_index()
prophetTest.drop(FEATURES, axis=1, inplace=True)
prophetTest.rename(columns={"DateTime": "ds", "Consumption": "y"}, inplace=True)

prophetTest.tail()

prophetModel = Prophet()
prophetModel.fit(prophetTrain)

prophetPrediction = prophetModel.predict(prophetTest)

yRealProphet = test["Consumption"]
yPredictedProphet = prophetPrediction["yhat"]

print(f"Prophet percentage error with no features: {meanAbsolutErrorAaPercentage(yRealProphet, yPredictedProphet):.4f}")

from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate RMSE
rmseNoFeatures = sqrt(mean_squared_error(yRealProphet, yPredictedProphet))
print('RMSE no features:')
print(rmseNoFeatures)
print(f"Prophet percentage error with no features: {meanAbsolutErrorAaPercentage(yRealProphet, yPredictedProphet):.4f}")
# Ensure yRealProphet and yPredictedProphet are numeric
yRealProphet_numeric = pd.to_numeric(yRealProphet, errors='coerce')
yPredictedProphet_numeric = pd.to_numeric(yPredictedProphet, errors='coerce')
r2 = r2_score(yRealProphet_numeric, yPredictedProphet_numeric)
print(f"R² Score: {r2}")

ax = train[TARGET].plot(figsize=(15,5), label='Train')
test[TARGET].plot(ax=ax, label='Test', style="k-")
prophetPrediction.set_index("ds")["yhat"].plot(ax=ax, style=".", color="g", label='Prediction')
ax.legend()
ax.set_title("Data and prediction")
# plt.show()



import matplotlib.pyplot as plt

# Initialize the Prophet model
prophetModel = Prophet()

# Add additional regressors
for feature in FEATURES:
    prophetModel.add_regressor(feature)

# Prepare training data with additional regressors
prophetTrain = train.reset_index().rename(columns={"DateTime": "ds", TARGET: "y"})

# Fit the model
prophetModel.fit(prophetTrain)

prophetTest = test.reset_index()
prophetTest.rename(columns={"DateTime": "ds", TARGET: "y"}, inplace=True)

# Use the model to make a forecast
prophetPrediction = prophetModel.predict(prophetTest)

# Calculate error
yRealProphet = test[TARGET]
yPredictedProphet = prophetPrediction["yhat"]
print(f"Prophet percentage error: {meanAbsolutErrorAaPercentage(yRealProphet, yPredictedProphet):.4f}")

from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate RMSE
rmseFeatures = sqrt(mean_squared_error(yRealProphet, yPredictedProphet))
print('RMSE after considering features:')
print(rmseFeatures)
print(f"Prophet percentage error: {meanAbsolutErrorAaPercentage(yRealProphet, yPredictedProphet):.4f}")
# Ensure yRealProphet and yPredictedProphet are numeric
yRealProphet_numeric = pd.to_numeric(yRealProphet, errors='coerce')
yPredictedProphet_numeric = pd.to_numeric(yPredictedProphet, errors='coerce')
r2 = r2_score(yRealProphet_numeric, yPredictedProphet_numeric)
print(f"R² Score: {r2}")

ax = train[TARGET].plot(figsize=(15,5), label='Train')
test[TARGET].plot(ax=ax, label='Test', style="k-")
prophetPrediction.set_index("ds")["yhat"].plot(ax=ax, style=".", color="g", label='Prediction')
ax.legend()
ax.set_title("Data and prediction")
# plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    non_zero_mask = denominator != 0
    return 100 * np.mean(np.abs(y_pred[non_zero_mask] - y_true[non_zero_mask]) / denominator[non_zero_mask])

# Ensure yRealProphet and yPredictedProphet are numeric
yRealProphet_numeric = pd.to_numeric(yRealProphet, errors='coerce')
yPredictedProphet_numeric = pd.to_numeric(yPredictedProphet, errors='coerce')

# Calculate the metrics
mae = mean_absolute_error(yRealProphet_numeric, yPredictedProphet_numeric)
mse = mean_squared_error(yRealProphet_numeric, yPredictedProphet_numeric)
rmse = np.sqrt(mse)
mape_score = mape(yRealProphet_numeric, yPredictedProphet_numeric)
smape_score = smape(yRealProphet_numeric, yPredictedProphet_numeric)
r2 = r2_score(yRealProphet_numeric, yPredictedProphet_numeric)

# Print the metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape_score}%")
print(f"SMAPE: {smape_score}%")
print(f"R² Score: {r2}")



# Combine train and test sets
full_data = pd.concat([train, test]).reset_index().rename(columns={"DateTime": "ds", TARGET: "y"})

prophetModel = Prophet()
for feature in FEATURES:
    prophetModel.add_regressor(feature)

# Fit the model with the full dataset
prophetModel.fit(full_data)

# Create a future DataFrame for 1 year beyond the last date in the test set
last_date = test.index[-1]
future_dates = prophetModel.make_future_dataframe(periods=366 * 24, freq='H', include_history=False)

# Add expected values for the additional regressors for the future dates
future_dates['hour'] = future_dates['ds'].dt.hour
future_dates['day_of_week'] = future_dates['ds'].dt.dayofweek
future_dates['quarter'] = future_dates['ds'].dt.quarter
future_dates['month'] = future_dates['ds'].dt.month
future_dates['year'] = future_dates['ds'].dt.year
future_dates['day_of_year'] = future_dates['ds'].dt.dayofyear
future_dates['holiday'] = future_dates['ds'].apply(lambda x: x in holidays)
future_dates['is_weekend'] = future_dates['ds'].dt.dayofweek.isin([5, 6])

# Predict consumption for the future datesa
forecast = prophetModel.predict(future_dates)

# Predict consumption for the future dates
forecast = prophetModel.predict(future_dates)


print(forecast[['ds', 'yhat']].describe())
print(forecast[['ds', 'yhat']].head())
print(forecast[['ds', 'yhat']].tail())

# API endpoint to retrieve forecast for specific days
@app.route('/daily_forecast', methods=['GET'])
def daily_forecast():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date', start_date)
    print(f"Start Date: {start_date}, End Date: {end_date}")  # Debugging print statement
    daily_forecast = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]
    print(daily_forecast)  # Debugging print statement
    return jsonify(daily_forecast[['ds', 'yhat']].to_dict(orient='records'))


# API endpoint to retrieve forecast for an entire year
@app.route('/yearly_forecast', methods=['GET'])
def yearly_forecast():
    year = request.args.get('year')
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    yearly_forecast = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]
    return jsonify(yearly_forecast[['ds', 'yhat']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)