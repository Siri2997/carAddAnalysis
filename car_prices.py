'''
Purpose: Predicting the car prices using machine learning Model
Name: Sireesha Maguluri
Date: 10/18/2023
'''


import pyreadr as pyr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Load data from RData file
carAd_file = pyr.read_r("car_ads_fp.RData")
carAd = carAd_file['carAd']
carAd = pd.DataFrame(carAd)
# Check the last 5 rows
carAd.tail()
# Select the top 50 models
mods = carAd["Genmodel"].value_counts().index.tolist()[:50]
# Select the top 6 colors
cols = carAd["Color"].value_counts().index.tolist()[:6]
#creating a data frame with the name of car_Df for carAd
car_Df = pd.DataFrame(carAd)
car_Df.head()

# Copying Car DF
car_Df_copy = car_Df.copy()
car_Df.shape

#Data Preprocessing
# check data for duplicates
print("Number of duplicates: ", sum(car_Df.duplicated()))

# check data for missing values
print("Number of missing values: ", car_Df.isnull().sum().sum())

# show columns with missing values with their percentage(XX.XX%) 
print(car_Df.isnull().sum()[car_Df.isnull().sum() > 0] / car_Df.shape[0] * 100)

car_Df.columns
car_Df.describe()

# Apply label encoding to 'genmodel' and 'color'
car_Df['genmodel_encoded'] = label_encoder.fit_transform(car_Df['Genmodel'])
car_Df['color_encoded'] = label_encoder.fit_transform(car_Df['Color'])

# change manufacture_year to int
car_Df['Adv_year'] = car_Df['Adv_year'].astype('int64')

car_Df['Price'] = pd.to_numeric(car_Df['Price'], errors='coerce')
car_Df = car_Df.dropna(subset=['Price'])
car_Df['Price'] = car_Df['Price'].astype(int)

car_Df['Price'].fillna(-1, inplace=True)
car_Df['Price'] = car_Df['Price'].astype(int)

# One-hot encode 'genmodel' and 'color'
car_Df = pd.get_dummies(car_Df, columns=['Genmodel', 'Color'], prefix=['Genmodel', 'Color'], drop_first=True)

# get numeric columns with missing values
num_cols = car_Df.select_dtypes(include=['int64', 'float64']).columns

# fill missing values with mean
for col in num_cols:
    car_Df[col].fillna(car_Df[col].mean(), inplace=True)

# get categorical columns with missing values
cat_cols = car_Df.select_dtypes(include=['object']).columns

# fill missing values with mode
for col in cat_cols:
    car_Df[col].fillna(car_Df[col].mode()[0], inplace=True)

# check data for missing values
#print("Number of missing values: ", car_Df.isnull().sum().sum())

#print(car_Df.info())

# summary statistics for numeric columns
print(car_Df.describe())

# summary statistics for categorical columns
#print(car_Df.describe(include=['object']))

import numpy as np
# Calculate the Z-score of each value in the numerical columns
z_scores = car_Df[['Adv_year', 'Adv_month', 'Reg_year', 'Price', 'genmodel_encoded', 'color_encoded']].apply(lambda x: (x - x.mean()) / x.std())

# Identify outliers as values with a Z-score greater than or less than 3
outliers = np.where(np.abs(z_scores) > 3)

# Display the outliers
print("Outliers in the numerical columns:")
print(car_Df.iloc[outliers[0]])

# Remove the outliers
car_Df = car_Df.drop(car_Df.index[outliers[0]])

# reset index
car_Df.reset_index(drop=True, inplace=True)

# Fetching the correlation matrix
corr = car_Df[['Adv_year', 'Adv_month', 'Reg_year', 'Price', 'genmodel_encoded', 'color_encoded']].corr()
# Create the heatmap
sns.heatmap(corr, annot=True, cmap='viridis')

# Show the plot
plt.show()

# Statistical results
print (car_Df[['Adv_year', 'Adv_month', 'Reg_year', 'Price', 'genmodel_encoded', 'color_encoded']].describe().round().astype(int))

column = car_Df['Price']
normalized_column = (column - column.mean()) / column.std()
car_Df['Price'] = normalized_column

#Here I am creating Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# create sample ML features
features = ['Adv_year', 'Adv_month', 'Reg_year', 'Price', 'genmodel_encoded', 'color_encoded']
target = ['Price']

# with data having 3million rows, we will take a sample of 100000 rows
car_Dfsample = car_Df.sample(n=100000, random_state=42) 

X = car_Dfsample[features].values
y = car_Dfsample[target].values

# Ensure y is a 1D array
y = car_Dfsample[target].values.ravel()
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
rfr_mse = mean_squared_error(y_test, rfr_pred)
rfr_r2 = r2_score(y_test, rfr_pred)
print('Random Forest Regressor MSE: ', rfr_mse)
print("R² on training data =: ", rfr_r2)


#testing model
pred = rfr.predict(X_test)
print("R\N{SUPERSCRIPT TWO}=", r2_score(y_test, pred))
# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_mse = mean_squared_error(y_test, gbr_pred)
gbr_r2 = r2_score(y_test, gbr_pred)
print('Gradient Boosting Regressor MSE: ', gbr_mse)
print("R² on test data:", gbr_r2)

# Scatter plot for actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rfr_pred, alpha=0.5)
plt.title('Random Forest Regressor: Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()