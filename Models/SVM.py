# Support Vector Machine

# importing the data
import pandas as pd
ads = pd.read_csv('Social_Network_Ads.csv')
x_var = ads.iloc[:, 2:4].values
y_var = ads.iloc[:, -1].values

# Scaling the data
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_var = x_scale.fit_transform(x_var)

# Splitting data into train and test sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# Fitting the SVM Model
from sklearn.svm import SVC
ads_svm = SVC(kernel = 'linear', random_state = 0)
ads_svm.fit(x_train, y_train)

# predicting on x_test
ads_pred = ads_svm.predict(x_test)

# building a confusion matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)