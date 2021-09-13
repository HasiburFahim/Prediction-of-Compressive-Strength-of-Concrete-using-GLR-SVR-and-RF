from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# RFR
df = pd.read_excel('C:/Users/Asus/PycharmProjects/CSE498R/compressive_strength.xlsx')
data = df.iloc[:, 0:8]
X = data.iloc[:, 1:].values
y = data.iloc[:, :1].values
y = np.array(y).reshape(-1, 1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1, 7))
y = sc_y.fit_transform(y.reshape(-1, 1))
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=True, max_depth=60, random_state=42)
regressor.fit(X_trn.reshape(-1, 7), y_trn.reshape(-1, 1).ravel())
y_predd = regressor.predict(X_tst)
y_predds = sc_y.inverse_transform(y_predd)


print("MSE")
print(mean_squared_error(y_tst, y_predd))
print("RMSE")
print(mean_squared_error(y_tst, y_predd, squared=False))
print("MAE")
print(mean_absolute_error(y_tst, y_predd))
show = pd.DataFrame({'Real Values': sc_y.inverse_transform(y_tst.reshape(-1)), 'Predicted Values': y_predd})
print(show)

plt.scatter(y_tst, y_predd)
plt.plot([y_tst.min(), y_tst.max()], [y_tst.min(), y_tst.max()], 'k--', 'lw=2')
plt.title('Random Forest Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

sns.displot(y_predd, label="Predicted")
sns.displot(y_tst, label="Real")
plt.show()