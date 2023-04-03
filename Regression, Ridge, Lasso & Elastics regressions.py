from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()

dataset

print(dataset.DESCR)

dataset.feature_names

import pandas as pd

df = pd.DataFrame(dataset.data,columns = dataset.feature_names)
df.head()

df['Price'] = dataset.target

df.head()

df.describe()

df.describe().T

df.isnull().sum()

import seaborn as sns

sns.pairplot(df)

df_copy = df.sample(frac = 0.25)

df_copy.shape

sns.pairplot(df_copy)

df.head()

"""## Divide the data into independent and independent"""

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(X)
print(y)

"""##Divide the data into train and test"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 36)

print(X_train,X_test,y_train,y_test)

X_train.head()

X.shape

X_train.shape,X_test.shape

"""##Feature Scaling -- Stadardization"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

"""### scaler.fit(X_train) --> only on train data not for test data
### scaler.tranform(X_test)  --> we apply on both test and train as well 
"""

scaler.fit(X_train)

scaler.transform(X_train)

# if you want use both fitand transform in one code then we apply

X_train = scaler.fit_transform(X_train)

X_train

X_test = scaler.transform(X_test)
X_test

"""### Model Traing """

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

"""### Model fit"""

regression.fit(X_train,y_train)

"""### Coefficients and Intercept """

regression.coef_

regression.intercept_

"""### Prediction"""

# Prediction
y_pred = regression.predict(X_test)

y_pred

"""### MSE, MAE and RMSE"""

from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
mse = mean_squared_error(y_test,y_pred)
print(mse)
mae = mean_absolute_error(y_test,y_pred)
print(mae)
print(np.sqrt(mse))

"""### Accuracy R^2 and Ajusted R^2"""

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)
r2

# disply adjusted R-squared

(1-(1-r2))*(len(y)-1)/(len(y)-X.shape[1]-1)

"""# Ridge

"""

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=30.0)  # alpha = 30.0 --> I choose randomly to check the variation in accuracy

ridge.fit(X_train,y_train)

y_pred = ridge.predict(X_test)

import numpy as np
mse = mean_squared_error(y_test,y_pred)
print(mse)
mae = mean_absolute_error(y_test,y_pred)
print(mae)
print(np.sqrt(mse))

"""# Lasso"""

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 30.0)
lasso.fit(X_train,y_train)

y_pred = lasso.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)
mae = mean_absolute_error(y_test,y_pred)
print(mae)
print(np.sqrt(mse))

"""# Elastic"""

from sklearn.linear_model import ElasticNet

elasticnet = ElasticNet(alpha = 30.0)
elasticnet.fit(X_train,y_train)

y_pred = elasticnet.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)
mae = mean_absolute_error(y_test,y_pred)
print(mae)
print(np.sqrt(mse))

df_copy.corr()

