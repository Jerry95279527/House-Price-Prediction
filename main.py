import pandas as pd

# Access data
train = pd.read_csv("C:\\python自主學習\\House-Price-Prediction\\train.csv")
test = pd.read_csv("C:\\python自主學習\\House-Price-Prediction\\test.csv")

# Target column
target = train['SalePrice']

train = train.drop(columns = ['Id','SalePrice'])
test = test.drop(columns = 'Id')

# Find missing value
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

# Remove columns which over 50% missing values are in the column
# df = df.loc[:, df.isnull().mean() < 0.5]
train = train.drop(columns = ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC'])
test = test.drop(columns = ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC'])

# Scale the continuous variable before processing KNN
import numpy as np
from sklearn.preprocessing import StandardScaler

numerical_columns_train = train.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_test = test.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns_train:
    scaler = StandardScaler()
    
    # Save the mask of NaN values
    train_nan_mask = train[column].isna()
    test_nan_mask = test[column].isna()
    
    # Temporarily fill NaN with the mean
    train_no_nan = train[column].fillna(train[column].mean()).values.reshape(-1, 1)
    test_no_nan = test[column].fillna(test[column].mean()).values.reshape(-1, 1)
    
    # StandardScaler
    train_scaled = scaler.fit_transform(train_no_nan).reshape(-1)
    test_scaled = scaler.transform(test_no_nan).reshape(-1)
    
    # Assign the scaled values back to the DataFrame
    train[column] = train_scaled
    test[column] = test_scaled
    
    # Restore the NaN values with np.nan to maintain float type
    train.loc[train_nan_mask, column] = np.nan
    test.loc[test_nan_mask, column] = np.nan


print(train.dtypes)

# Find categorical column
categorical_columns_train = train.select_dtypes(include=['object']).columns
categorical_columns_test = train.select_dtypes(include=['object']).columns

# Fill nan in categorical columns by using 'missing' string
for column in categorical_columns_train:
    train[column].fillna('missing',inplace = True)
    test[column].fillna('missing',inplace = True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for column in categorical_columns_train:
    train[column] = le.fit_transform(train[column])
    test[column] = le.fit_transform(test[column])
    

print(train.dtypes)

# KNN
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

neighbors_settings = range(3, 10)

# using cross validation to decide the number of n_neighbor
best_score = float('-inf')
best_n_neighbors = 0

for n_neighbors in neighbors_settings:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    train_imputed = imputer.fit_transform(train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, train_imputed, target, cv=5, scoring='neg_mean_squared_error')
    mean_score = scores.mean()
    print(mean_score)
    if mean_score > best_score:
        best_score = mean_score
        best_n_neighbors = n_neighbors

print(f"最佳的 n_neighbors 數量是: {best_n_neighbors}")

# fill nan
imputer = KNNImputer(n_neighbors=best_n_neighbors)
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# convert to dataframe
train = pd.DataFrame(train_imputed, columns=train.columns)
test = pd.DataFrame(test_imputed, columns=test.columns)

