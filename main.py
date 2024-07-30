import pandas as pd
import numpy as np
from scipy import stats


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
categorical_columns_test = train.select_dtypes(include=['object']).columns

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

#訓練模型?

#