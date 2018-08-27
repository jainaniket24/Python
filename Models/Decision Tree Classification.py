# Decision Tree Classification

# importing dataset
import pandas as pd
ads = pd.read_csv('Social_Network_ads.csv')
x_var = ads.iloc[:, 2:4].values
y_var = ads.iloc[:, -1].values

# splitting into train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.25, random_state = 0)

# fitting the decision tree model
from sklearn.tree import DecisionTreeClassifier
ads_cart = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
ads_cart.fit(x_train, y_train)

# predicting the results
ads_pred = ads_cart.predict(x_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
ads_confMat = confusion_matrix(y_test, ads_pred)
