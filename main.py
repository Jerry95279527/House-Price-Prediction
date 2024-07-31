import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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

#fill missing value by median in that numerical column
numerical_columns_train = train.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_test = test.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns_train:
    train[column] = np.where(train[column].isnull(),np.nanmedian(train[column]),train[column])
    test[column] = np.where(test[column].isnull(),np.nanmedian(test[column]),test[column])

#fill missing value by mode in that categorical column
categorical_columns_train = train.select_dtypes(include=['object']).columns
categorical_columns_test = test.select_dtypes(include=['object']).columns

for column in categorical_columns_train:
    train[column] = np.where(train[column].isnull(),stats.mode(train[column])[0],train[column])
    test[column] = np.where(test[column].isnull(),stats.mode(test[column])[0],test[column])


# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for column in categorical_columns_train:
    train[column] = le.fit_transform(train[column])
    test[column] = le.transform(test[column])

print(train.dtypes)


#calculate the correlation coefficient of each column
train['SalePrice'] = target
correlation_matrix = train.corr()
correlation_with_target = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)

#select the column which correlation coefficient are greater than 0.5
selected_features = correlation_with_target[correlation_with_target > 0.5].index

#drop the columns which correlation coefficient are less than 0.5
for columns in train:
    if columns not in selected_features:
        train = train.drop(columns,axis = 1)
        test = test.drop(columns,axis = 1)

train = train.drop(columns = ['SalePrice'])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_n_estimators = 0
best_RMSE = float('inf')

for i in range(100, 501, 50):
    # Train RandomForestRegressor
    clf = RandomForestRegressor(n_estimators=i, random_state=0)
    clf.fit(X_train_scaled, y_train)

    # Evaluate model
    train_score = clf.score(X_train_scaled, y_train)
    test_pred = clf.predict(X_test_scaled)
    test_score = mean_squared_error(y_test, test_pred, squared=False)

    print(f"n_estimators = {i}")
    print(f"隨機森林訓練資料集正確率 = {train_score}") 
    print(f"隨機森林測試資料集RMSE = {test_score}")  #31060.5008

    # Update best_n_estimators and best_RMSE
    if test_score < best_RMSE:
        best_RMSE = test_score
        best_n_estimators = i

final_clf = RandomForestRegressor(n_estimators=best_n_estimators, random_state=0)
final_clf.fit(X_train_scaled, y_train)

#predict
test_scaled = scaler.transform(test)
final_test_pred = final_clf.predict(test_scaled)


# Gradient Boosting Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

gbr = GradientBoostingRegressor(random_state=0)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"Best Gradient Boosting Params: {grid_search.best_params_}")
print(f"Best Gradient Boosting RMSE: {np.sqrt(-grid_search.best_score_)}") #34105.0693

# Train final Gradient Boosting model
final_gbr_clf = grid_search.best_estimator_

# Predict with Gradient Boosting
final_gbr_test_pred = final_gbr_clf.predict(test_scaled)
with open('house_predict_rf.csv', 'w') as f: 
    f.write('id,SalePrice\n') 
    for i in range(len(final_test_pred)): 
        f.write(str(i + 1461) + ',' + str(float(final_test_pred[i])) + '\n')
        