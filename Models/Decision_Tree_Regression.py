# Decision Tree Regression

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data set
salary = pd.read_csv('Position_Salaries.csv')
x_var = salary.iloc[:, 1:2].values
y_var = salary.iloc[:, 2].values

# Fitting the decision tree model
from sklearn.tree import DecisionTreeRegressor

salary_tree = DecisionTreeRegressor(random_state=0)
salary_tree.fit(x_var, y_var)

# predicting the salary at level = 6.5
y_pred = salary_tree.predict(6.5)

# visualizing the results in higher resolution as decision tree regression 
# is discontinuous 
plt.scatter(x_var, y_var, color = 'red')
x_grid = np.arange(min(x_var), max(x_var), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.plot(x_grid, salary_tree.predict(x_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

