# Polinomial Regression

# reading data
import pandas as pd

salary = pd.read_csv('Position_Salaries.csv')
# only including encoded leveles column as the x-variable
x_var = salary.iloc[:, 1:2].values
y_var = salary.iloc[:, 2].values

# fitting the linear regression model to the data
from sklearn.linear_model import LinearRegression
salary_lin = LinearRegression()
salary_lin.fit(x_var, y_var)

# fitting the polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x_var)
salary_poly = LinearRegression()
salary_poly.fit(x_poly, y_var)

# visualizing results of linear regression
import matplotlib.pyplot as plt
plt.scatter(x_var, y_var, color = 'red')
plt.plot(x_var, salary_lin.predict(x_var), color = 'blue')
plt.title('Linear Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing results of polynomial regression
plt.scatter(x_var, y_var, color = 'red')
plt.plot(x_var, salary_poly.predict(poly_reg.fit_transform(x_var)), color = 'blue')
plt.title('Polynomial Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# adding continuity to curve - removing straight lines between x-points
import numpy as np
x_grid = np.arange(min(x_var), max(x_var), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x_var, y_var, color = 'red')
plt.plot(x_grid, salary_poly.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Polynomial Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting value at level 6.5 from linear model
salary_lin.predict(6.5) # 330378

# predicting value at level 6.5 from polynomial model
salary_poly.predict(poly_reg.fit_transform(6.5)) # 133259

