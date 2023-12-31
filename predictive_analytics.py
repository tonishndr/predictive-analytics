# -*- coding: utf-8 -*-
"""mltp_predictive_analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oKIES_QYOHp9KxdOmdmYEAbjeepAxkDI

##### **Data Explorations**

Hal pertama yang dilakukan adalah mengimport librari umum untuk keperluan analisis.
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

"""Memuat dataset menjadi dataframe."""

# Load Data
or_df = pd.read_csv('/Users/tonisuhendar/One-Hour/learn-ds/dataset/online-retail-transactions/online_retail.csv')
or_df

"""Menampilkan informasi terkait dataset."""

# Get info
or_df.info()

"""Melakukan pengecekan nilai yang hilang."""

# Check Missing Values
or_df.isna().sum()

"""Membuat dataframe sebagai representasi nilai yang hilang dalam ukuran persentase supaya lebih mudah dipahami."""

# Percentage of Missing Values
total_number_of_rows = len(or_df)
miss_description = or_df['Description'].isna().sum()
miss_customerid = or_df['CustomerID'].isna().sum()
miss_desc_cust = or_df.isna().any(axis=1).sum()

percent_miss_description = round(miss_description / total_number_of_rows * 100, 2)
percent_miss_customerid = round(miss_customerid / total_number_of_rows* 100, 2)
percent_miss_desc_cust = round(miss_desc_cust / total_number_of_rows* 100, 2)


missing_value = pd.DataFrame({'Count': [total_number_of_rows, miss_description, miss_customerid, miss_desc_cust],
                              'Percent': [100, percent_miss_description, percent_miss_customerid, percent_miss_desc_cust]},
                             index=['Total Rows', 'Description', 'Customer ID', 'Total Missing Values'])

missing_value['Percent'] = missing_value['Percent'].apply(lambda x: "{:.2f}%".format(x))

missing_value

"""Menangani nilai yang ngilang pada variabel CustomerID dan Description dengan metode pengisian data."""

# Handles Missing Values ​​from CustomerID by filling with 'Guest' values
or_df['CustomerID'].fillna(value=0, inplace=True)
or_df['CustomerID'] = or_df['CustomerID'].astype('int')
or_df['CustomerID'] = or_df['CustomerID'].replace(0, 'Guest')
or_df['CustomerID'] = or_df['CustomerID'].astype('str')

# Handles Missing Values ​​from Description by filling with 'No Descriptions' values
or_df['Description'].fillna(value='No Descriptions', inplace=True)

"""Memvalidasi hasil daripada penganganan nilai yang hilang."""

# Percentage of Missing Values
total_number_of_rows = len(or_df)
miss_description = or_df['Description'].isna().sum()
miss_customerid = or_df['CustomerID'].isna().sum()
miss_desc_cust = or_df.isna().any(axis=1).sum()

percent_miss_description = round(miss_description / total_number_of_rows * 100, 2)
percent_miss_customerid = round(miss_customerid / total_number_of_rows* 100, 2)
percent_miss_desc_cust = round(miss_desc_cust / total_number_of_rows* 100, 2)


missing_value = pd.DataFrame({'Count': [total_number_of_rows, miss_description, miss_customerid, miss_desc_cust],
                              'Percent': [100, percent_miss_description, percent_miss_customerid, percent_miss_desc_cust]},
                             index=['Total Rows', 'Description', 'Customer ID', 'Total Missing Values'])

missing_value['Percent'] = missing_value['Percent'].apply(lambda x: "{:.2f}%".format(x))

missing_value

"""Melakukan pengecekan data duplikat."""

# Check for Duplicate Data
or_df.duplicated().sum()

"""Menampilkan data data yang duplikat."""

# Duplicate Data
dup = or_df[or_df.duplicated(keep=False)]
dup

"""Menangadi data duplikat dengan menghapusnya."""

# Drop Duplicate Data
or_df = or_df.drop_duplicates(keep='first')
or_df.duplicated().sum()

"""Menambahkan variabel TotalPrices pada dataframe."""

# Add new column TotalPrices = Quantity * UnitPrice
or_df['TotalPrices'] = or_df['Quantity'] * or_df['UnitPrice']

# Re-order Feature
cl_df = or_df[['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'TotalPrices']]
cl_df = cl_df.reset_index(drop=True)
cl_df

"""Melihat informasi data."""

cl_df.info()

"""Merubah tipe data pada variabel InvoiceDate."""

# Change Dtype
cl_df['InvoiceDate'] = pd.to_datetime(cl_df['InvoiceDate'])
cl_df.info()

"""##### **Data Visualization**

Membuat visualisasi data berdasarkan pertanyaan umum

1. Negara mana yang memiliki transaksi terbanyak?
"""

# Top 10 Sales by Country

sales_by_country = cl_df.groupby('Country')['TotalPrices'].sum().sort_values(ascending=False)
sales_by_country = sales_by_country.head(10)
 
ax = sales_by_country.plot.bar()

plt.bar(sales_by_country.index, sales_by_country.values)

ax.set_xlabel('Country')
ax.set_ylabel('Total')
ax.set_title('Top 10 Sales by Country')

# Add the count values to the bars
for i, v in enumerate(sales_by_country.values):
    ax.text(i, v + 100000, str(v), ha='center')
plt.subplots_adjust(right=2.5, top=1)

"""2. Produk apa yang paling laris?"""

# Top 5 Sales by Product

sales_by_product = cl_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
sales_by_product = sales_by_product.head(5)
 
ax = sales_by_product.plot.bar()

plt.bar(sales_by_product.index, sales_by_product.values)

ax.set_xlabel('Products')
ax.set_ylabel('Count')
ax.set_title('Top 5 Sales by Product')

# Add the count values to the bars
for i, v in enumerate(sales_by_product.values):
    ax.text(i, v + 500, str(v), ha='center')
plt.subplots_adjust(right=2.5, top=1)

"""3. Pelanggan mana yang paling banyak berbelanja?"""

# Top 5 Buyer

monthly_data = cl_df.groupby('CustomerID')['TotalPrices'].sum().sort_values(ascending=False)
monthly_data = monthly_data.head(5)
 
ax = monthly_data.plot.bar()

plt.bar(monthly_data.index, monthly_data.values)

ax.set_xlabel('Customer ID')
ax.set_ylabel('Total')
ax.set_title('Top 5 Buyer')

# Add the count values to the bars
for i, v in enumerate(monthly_data.values):
    ax.text(i, v + 3000, str(v), ha='center')
plt.subplots_adjust(right=2.5, top=1)

"""4. Bagaimana siklus penjualan setiap bulannya?"""

# Group the data by month and sum the Quantity and TotalPrices values for each month
monthly_data = cl_df.resample('M', on='InvoiceDate').sum()

# Create a line chart of the monthly Quantity and TotalPrices values
ax = monthly_data.plot(y=['TotalPrices'], kind='line')
ax.set_xlabel('Month')
ax.set_ylabel('Value')
ax.set_title('Monthly Income')

plt.subplots_adjust(right=2.5, top=1)

"""5. Berapa rata-rata siklus penjualan setiap harinya?"""

# Group the data by day and average the Quantity and TotalPrices values for each month
daily_data = cl_df.groupby(cl_df['InvoiceDate'].dt.day)['TotalPrices'].mean()

# Create a line chart of the monthly Quantity and TotalPrices values
ax = daily_data.plot(y=['TotalPrices'], kind='line')
ax.set_xlabel('Days')
ax.set_ylabel('Value')
ax.set_title('Average Daily Income')

plt.subplots_adjust(right=2.5, top=1)

"""6. Bagaimana trend penjualan harian dalam satu tahun?"""

import plotly.express as px

# Plot data by day and sum the 'Quantity' and 'TotalPrices' column
sumbyday_cl_df = cl_df.groupby(pd.Grouper(key='InvoiceDate', freq='D')).sum().reset_index()

# Create the plot
fig = px.line(sumbyday_cl_df, x='InvoiceDate', y=['TotalPrices'], title='Sum by Days - Total Prices')

# Add x-axis and y-axis labels
fig.update_layout(xaxis_title='Invoice Date', yaxis_title='Value')

trend_prices = np.polyfit(sumbyday_cl_df.index, sumbyday_cl_df['TotalPrices'], 1)

fig.add_scatter(x=sumbyday_cl_df['InvoiceDate'], y=np.polyval(trend_prices, sumbyday_cl_df.index), mode='lines', name='Total Prices Trend')

# Display the plot
fig.show()

"""#### **Build Dataset For Modeling**

Membuat dataframe baru untuk kebutukan pemodelan.
"""

dataset = cl_df[['InvoiceDate', 'TotalPrices']]
dataset

"""Melihat informasi data."""

dataset.info()

"""Melakukan visualisasi data untuk melihat hubungan antara variabel."""

# Plot
fig, axs = plt.subplots(figsize=(15,5))
dataset.plot('InvoiceDate', 'TotalPrices', ax=axs)

plt.tight_layout()

"""Menampilkan ringkasan statistik deskriptif dari dataset."""

dataset.describe()

"""Mendeteksi outliers dengan visualisasi BoxPlot."""

# Outlier Detection with BoxPlot
fig, axs = plt.subplots(figsize=(15, 5))

sns.boxplot(x=dataset['TotalPrices'], ax=axs)

fig.suptitle('Outliers detections with BoxPlot')
plt.tight_layout()

"""Melihat hasil minimum dan maksimun daripada metode IQR untuk menangasi outliers."""

# Convert Dtype
dataset['InvoiceDate'] = dataset['InvoiceDate'].astype('str')

# Handling Outliers
Q1 = (dataset[['TotalPrices']]).quantile(0.25)
Q3 = (dataset[['TotalPrices']]).quantile(0.75)
IQR = Q3 - Q1

maximun = Q3 + (1.5 * IQR)
minimum = Q1 - (1.5 * IQR)

dataset_outlier = pd.DataFrame({'Minimum Value': minimum, 
                                'Maximum Value': maximun,}, 
                                index=['TotalPrices'])
dataset_outlier

"""Menangani outliers dengan normalisasi data berdasarkan metode IQR."""

more_than = (dataset > maximun)
lower_than = (dataset < minimum)

dataset = dataset.mask(more_than, maximun, axis=1)
dataset = dataset.mask(lower_than, minimum, axis=1)

"""Memvalidasi persebaran data setelah menangani masalah outliers dengan metode IQR, visualisasi dengan BoxPlot."""

# Outlier Detection with BoxPlot
fig, axs = plt.subplots(figsize=(15, 5))
sns.boxplot(x=dataset['TotalPrices'], ax=axs)

fig.suptitle('Outliers detections with BoxPlot')
plt.tight_layout()

"""Memvisualisasikan persebaran dataset setelah di normalisasi."""

# Convert Dtype 
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])

# Plot
fig, axs = plt.subplots(figsize=(15,5))
dataset.plot('InvoiceDate', 'TotalPrices', ax=axs)

plt.tight_layout()

"""Melakukan visualisasi data dengan plot garis untuk melihat grafik penjualan harian."""

import plotly.express as px

# Plot data by day and sum the 'Quantity' and 'TotalPrices' column
sumbyday_dataset = dataset.groupby(pd.Grouper(key='InvoiceDate', freq='D')).sum().reset_index()

# Create the plot
fig = px.line(sumbyday_dataset, x='InvoiceDate', y=['TotalPrices'], title='Sum by Days - Total Prices')

# Add x-axis and y-axis labels
fig.update_layout(xaxis_title='Invoice Date', yaxis_title='Value')

# Display the plot
fig.show()

"""Melakukan visualisasi data dengan plot garis untuk melihat grafik penjualan bulanan."""

import plotly.express as px

# Plot data by month and sum the 'Quantity' and 'TotalPrices' column
sumbymonth_dataset = dataset.groupby(pd.Grouper(key='InvoiceDate', freq='M')).sum().reset_index()

# Create the plot
fig = px.line(sumbymonth_dataset, x='InvoiceDate', y=['TotalPrices'], title='Sum by Month - Total Prices')

# Add x-axis and y-axis labels
fig.update_layout(xaxis_title='Invoice Date', yaxis_title='Value')

# Display the plot
fig.show()

"""Menampilkan dataframe yang sudah dikalkulasikan berdasarkan data harian."""

sumbyday_dataset

"""Melakukan ektrasi fitur waktu untuk kebutuhan pemodelan supaya analisis bisa lebih detail."""

# Extract InvoiceDate Features
dataset['day'] = dataset['InvoiceDate'].dt.day
dataset['month'] = dataset['InvoiceDate'].dt.month
dataset['year'] = dataset['InvoiceDate'].dt.year
dataset['dayofweek'] = dataset['InvoiceDate'].dt.dayofweek
dataset['quarter'] = dataset['InvoiceDate'].dt.quarter
dataset['weekday'] = dataset['dayofweek'] < 5

dataset

"""Melihat informasi data."""

dataset.info()

"""#### **Modeling**

Tahap pemodelan, pertama mengimport librari yang dibutuhkan, menentukan nilai X dan y, menerapkan metode normalisasi data pada variabel y yaitu dengan MinMaxScaler, membagi data dengan metode TimeSeriesSplit, melakukan pengujian terhadap model.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define X and y
X = dataset[['day', 'month', 'year', 'dayofweek', 'quarter', 'weekday']].values
y = dataset['TotalPrices'].values

# Initialize scaler
scaler = MinMaxScaler()

X = X
y = scaler.fit_transform(y.reshape(-1, 1))

# Data Splitting
tscv = TimeSeriesSplit(n_splits=5)

# Initialize models
linear_reg = LinearRegression()
decision_tree = DecisionTreeRegressor()

# Perform Time Series Cross Validation
mae_linear_reg = []
mae_decision_tree = []
mse_linear_reg = []
mse_decision_tree = []
rmse_linear_reg = []
rmse_decision_tree = []


for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train and predict using Linear Regression
    linear_reg.fit(X_train, y_train)
    y_pred_linear_reg = linear_reg.predict(X_test)
    mse_linear_reg = mean_squared_error(y_test, y_pred_linear_reg)
    rmse_linear_reg = np.sqrt(mse_linear_reg)
    mae_linear_reg = mean_absolute_error(y_test, y_pred_linear_reg)

    # Train and predict using Decision Tree
    decision_tree.fit(X_train, y_train)
    y_pred_decision_tree = decision_tree.predict(X_test)
    mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
    rmse_decision_tree = np.sqrt(mse_decision_tree)
    mae_decision_tree = mean_absolute_error(y_test, y_pred_decision_tree)

print("MSE (Linear Regression): {:.3f}".format(mse_linear_reg))
print("RMSE (Linear Regression) {:.3f}".format(rmse_linear_reg))
print("MAE (Linear Regression): {:.3f}".format(mae_linear_reg))

print("MSE (Decision Tree): {:.3f}".format(mse_decision_tree))
print("RMSE (Decision Tree): {:.3f}".format(rmse_decision_tree))
print("MAE (Decision Tree): {:.3f}".format(mae_decision_tree))

"""#### **Evaluation**

Evaluasi dalam konteks pemodelan atau machine learning adalah proses mengevaluasi kinerja suatu model atau algoritma berdasarkan data yang telah dikumpulkan atau diuji.
Membuat dataframe untuk hasil pengujian model yang tekah dievaluasi metrik supaya lebih mudah di pahami.
"""

metrics_df = pd.DataFrame({
    'Linear Regression': [mse_linear_reg, rmse_linear_reg, mae_linear_reg],
    'Decision Tree': [mse_decision_tree, rmse_decision_tree, mae_decision_tree],
}, index=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'])
metrics_df['Linear Regression'] = metrics_df['Linear Regression'].round(3)
metrics_df['Decision Tree'] = metrics_df['Decision Tree'].round(3)
metrics_df

"""Membuat visualisasi model supaya lebih mudah dipahami."""

import matplotlib.pyplot as plt

metrics_df.plot.bar(rot=0)
plt.subplots_adjust(right=2.5, top=1)
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Comparison of Metrics: Linear Regression vs Decision Tree')
plt.legend(loc='lower right')
plt.show()