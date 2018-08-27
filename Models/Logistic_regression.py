# Logistic regression

# importing the  data
import pandas as pd
ads = pd.read_csv('Social_Network_Ads.csv')
x_var = ads.iloc[:, [2, 3]].values
y_var = ads.iloc[:, -1].values

# splitting the data set into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.fit_transform(x_test)

# fitting the logistic regression model
from sklearn.linear_model import LogisticRegression
ads_logistic = LogisticRegression(random_state = 0)
ads_logistic.fit(x_train, y_train)

# Predicting the test set results
ads_pred = ads_logistic.predict(x_test)

# Making confusion Matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)