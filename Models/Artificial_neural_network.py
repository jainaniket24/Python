# Churn Rate Estimation - Predicting if a customer would exit

# importing the dataset
import pandas as pd
churn = pd.read_csv('Churn_Modelling.csv')
x_var = churn.iloc[:, 3:-1].values
y_var = churn.iloc[:, -1].values

# encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_Geo = LabelEncoder()
x_var[:, 1] = label_encoder_Geo.fit_transform(x_var[:, 1])

label_encoder_Gender = LabelEncoder()
x_var[:, 2] = label_encoder_Gender.fit_transform(x_var[:, 2])

# creating dummy variables for country
dummy_Geo = OneHotEncoder(categorical_features=[1])
x_var = dummy_Geo.fit_transform(x_var).toarray()
x_var = x_var[:, 1:]

# splitting the data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_var, y_var, test_size = 0.20, random_state = 0)

# Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Making the Artificial Neural Network
# importing the Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
churn_model = Sequential()
# adding input and hidden layers
# input layer
churn_model.add(Dense(units = 6,
                      kernel_initializer = 'uniform',
                      activation = 'relu',
                      input_dim = 11))
# Hidden layer 1
churn_model.add(Dense(units = 6,
                      kernel_initializer = 'uniform',
                      activation = 'relu',
                      input_dim = 6)) 
# here adding input_dim is not required as model knows output from previous
# layer was 6 nodes which will act as input here

# output layer
churn_model.add(Dense(units = 1,
                      kernel_initializer = 'uniform',
                      activation = 'sigmoid'))

# Compiling the ANN
churn_model.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

# Fitting the ANN to the training set
churn_model.fit(x = x_train, y = y_train,
                batch_size = 10,
                nb_epoch = 100)

# making the predictions
y_pred = churn_model.predict(x_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
y_confMat = confusion_matrix(y_test, y_pred)




