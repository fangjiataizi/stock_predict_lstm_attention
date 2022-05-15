import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


data = pd.read_csv('data/btc.csv')
data=data[['close','open','high','low','vol']]
X = data.drop('close',axis=1)
Y = data['close']


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.33, random_state=42)

total_len=len(X)
X_train, y_train = X[:int(0.98 * total_len)], Y[:int(0.98 * total_len)]
X_test, y_test = X[int(0.98 * total_len):], Y[int(0.98 * total_len):]


mlp = MLPRegressor()

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)


print(f"均方误差(MSE)：{mean_squared_error(y_test, y_pred)}")
print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"测试集R^2：{r2_score(y_test, y_pred)}")


plt.plot(range(len(y_pred)),y_test,'-r',label='Close Price true')
plt.plot(range(len(y_pred)),y_pred,'-b',label='Close Price pred')

plt.legend()
plt.show()
