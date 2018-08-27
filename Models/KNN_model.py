# KNN Model

# importing the data
import pandas as pd
ads = pd.read_csv('Social_Network_Ads.csv')
x_var = ads.iloc[:, 2:4].values
y_var = ads.iloc[:, -1].values

# preprocessing the data
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_var = x_scale.fit_transform(x_var)

# splitting into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# fitting the KNN Model
from sklearn.neighbors import KNeighborsClassifier
ads_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
ads_knn.fit(x_train, y_train)

# predicting the results using KNN Model
ads_pred = ads_knn.predict(x_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)