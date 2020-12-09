#************************************************************************************
# Nhu Nguyen
# EE5321 – HW#4
# Filename: Homework_4_ANN.py
# Due: 11/11/20
#
# Objective: Using TensorFlow & Keras to classify the freight tracking 
# and tracing of the transportation data.
#The data has three classes (legs), 98 features, and 3,942 samples
# 
##************************************************************************************


#Importing all required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers



# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

#reading the dataset, creating dataframe
cargo = pd.read_csv (r'C:/Users/angel/Desktop/FALL 2020/EE/Assignment 4/c2k_data.csv', na_values=['?'])
print(cargo.shape)
cargo.info()

# fill missing values with mean column values
cargo.fillna(cargo.mean(), inplace=True)
# count the number of NaN values in each column
print(cargo.isnull().sum())

# Change to number values:
cargo = cargo.values

#Separating data & target information
X = cargo[:, 0:-1]
y = cargo[:,-1]


# Dataset parameters.
n_classes = 4 
n_features = 97

# Network parameters.
n_hidden1 = 65
n_hidden2 = 65
n_hidden3 = 15

# Build a model using Functional API
inputs = keras.Input(shape=(97,), name="digits")

# Add 3 hidden layers to the model
x1 = layers.Dense(n_hidden1, activation="sigmoid", name="dense_1")(inputs)
x2 = layers.Dense(n_hidden2, activation="sigmoid", name="dense_2")(x1)
x3 = layers.Dense(n_hidden3, activation="sigmoid", name="dense_3")(x2)

 
# Add the Output layer
outputs = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x3)

# Summary the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()



# Split the dataset into train, validation, and test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# Print trani, validation, and test data
print('\nTrain dataset: ', len(X_train))
print('Validation dataset: ', len(X_val))
print('Test dataset: ', len(X_test),)

#Scaling training, validation, and test input data:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_std = (X_train - mean_vals)/std_val
X_val_std = (X_val - mean_vals)/std_val
X_test_std = (X_test - mean_vals)/std_val
del X_train, X_test
print("\nTraining Scale: ", X_train_std.shape, y_train.shape)
print("Validation Scale: ", X_val_std.shape, y_val.shape)
print("Testing Scale: ", X_test_std.shape, y_test.shape)

# Compile the model
adam_optimizer= keras.optimizers.Adam(learning_rate = 0.01, beta_1 = 0.9)
model.compile(optimizer=adam_optimizer,
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# Print the fit training model:
print("\nFit model on training data")
model.fit(
    X_train_std,
    y_train,
    batch_size=60,
    epochs=20,
    validation_data=(X_val_std, y_val),
)


# Evaluate the model on the test data
print("\nEvaluate on train data")
train_loss, train_accuracy = model.evaluate(X_train_std, y_train, batch_size=120)
print("Train Loss: %.2f" % train_loss)
print("Train Accuracy: %.2f" % train_accuracy)

# Evaluate the model on the test data
print("\nEvaluate on test data")
test_loss, test_accuracy = model.evaluate(X_test_std, y_test, batch_size=120)
print("Test Loss: %.2f" % test_loss)
print("Test Accuracy: %.2f" % test_accuracy)

# printing out the training and test accuracy values in a text file
with open("HW4_ANN_output.txt", "w") as text_file:
   text_file.write(format("Train Accuracy: %.2f" % train_accuracy))
   text_file.write(format("\nTest Accuracy: %.2f" % test_accuracy))
    


