# Random Forest Classification Model

# Importing the data
import pandas as pd
ads = pd.read_csv('Social_Network_ads.csv')
x_var = ads.iloc[:, 2:4].values
y_var = ads.iloc[:, -1].values

# splitting into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# building the random forest model
from sklearn.ensemble import RandomForestClassifier
ads_rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', 
                                random_state = 0)
ads_rf.fit(x_train, y_train)

# predicting the values on test set
ads_pred = ads_rf.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)
