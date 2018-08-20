# SVR

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

salary = pd.read_csv('Position_Salaries.csv')
x_var = salary.iloc[:, 1:2].values
y_var = salary.iloc[:, 2:3].values

# Scaling the features
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_var = scaler_x.fit_transform(x_var)
y_var = scaler_y.fit_transform(y_var)

# fitting SVR to the data
from sklearn.svm import SVR

salary_svr = SVR(kernel = 'rbf')
salary_svr.fit(x_var, y_var)

# making prediction for value of x = 6.5 using SVR model
y_pred = salary_svr.predict(scaler_x.transform(np.array([[6.5]])))
y_pred = scaler_y.inverse_transform(y_pred)

# visualizing results
plt.scatter(x = x_var, y = y_var, color = 'red')
plt.plot(x_var, salary_svr.predict(x_var), color = 'blue')
plt.title('SVR Model Results')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# visulazing a smoother version of the predictions - by reducing intervals between x
x_grid = np.arange(min(x_var), max(x_var), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x = x_var, y = y_var, color = 'red')
plt.plot(x_grid, salary_svr.predict(x_grid), color = 'blue')
plt.title('SVR Model Results')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()