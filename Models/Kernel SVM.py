# Kernel SVM

# importing the data
import pandas as pd
ads = pd.read_csv('Social_Network_ads.csv')
x_var = ads.iloc[:, 2:4].values
y_var = ads.iloc[:, -1].values

# Scaling the data
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_var = x_scale.fit_transform(x_var)

# splitting the data into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# fitting the Kernel SVM
from sklearn.svm import SVC
ads_svm_kernel = SVC(kernel = 'rbf', random_state = 0)
ads_svm_kernel.fit(x_train, y_train)

# predicting the values
ads_pred = ads_svm_kernel.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)
