from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# SVR
df = pd.read_excel('C:/Users/Asus/PycharmProjects/CSE498R/compressive_strength.xlsx')
data = df.iloc[:, 0:8]
X = data.iloc[:, 1:].values
y = data.iloc[:, :1].values
y = np.array(y).reshape(-1, 1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1, 7))
y = sc_y.fit_transform(y.reshape(-1, 1))
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=None)


from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1, gamma='scale')
fit_svr = svr.fit(X_trn.reshape(-1, 7), y_trn.reshape(-1, 1).ravel())
y_predd = svr.predict(X_tst)
y_predds = sc_y.inverse_transform(y_predd)

r2 = fit_svr.score(X_tst, y_tst)
print("R2")
print(r2)
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
plt.title('Support Vector Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


sns.displot(y_predd, label='Predicted')
sns.displot(y_tst, label='Real')
plt.show()


param_grid = {
    'C': [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

optimal_params = GridSearchCV(SVR(),
                              param_grid,
                              cv=3,
                              # scoring='accuracy',
                              refit=True,
                              n_jobs=-1,
                              verbose=2)
optimal_params.fit(X_trn, y_trn.ravel())
print(optimal_params.best_params_)
