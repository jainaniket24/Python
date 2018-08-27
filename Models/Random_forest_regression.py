# Random Forest Regression

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data set
salary = pd.read_csv('Position_Salaries.csv')
x_var = salary.iloc[:, 1:2].values
y_var = salary.iloc[:, 2].values

# creating and fitting random forest model
from sklearn.ensemble import RandomForestRegressor
salary_rf = RandomForestRegressor(n_estimators = 300, random_state = 0)
salary_rf.fit(x_var, y_var)

# predicting the value at x = 6.5
salary_pred = salary_rf.predict(6.5)

# visualizing the results
plt.scatter(x_var, y_var, color = 'blue')
x_grid = np.arange(min(x_var), max(x_var), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.plot(x_grid, salary_rf.predict(x_grid), color = 'red')
plt.title('Random Forest Model')
plt.xlabel('Level')
plt.ylabel('Salary')
