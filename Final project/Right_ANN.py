
#************************************************************************************
# Nhu Nguyen
# EE5321 – Project
# Filename: Right_ANN.py
# Due: 12/09/20
#
# Objective:  
# 
#
# 
##************************************************************************************



#Importing all required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

#reading the dataset, creating dataframe
right_data = pd.read_csv (r'C:/Users/angel/Desktop/FALL 2020/EE/Project/Right Limb_2020_09_05.csv')
print(right_data.shape)
right_data.info()

# fill missing values with mean column values
right_data = right_data.dropna(axis=1, how='any')


# Change to number values:
right_data = right_data.values

#Separating data & target information
X = right_data[:, 2:-1]
y = right_data[:,-1]
print(X)
print(y)

# Parameters
n_classes = 2
n_features = 47
Epochs = 50
batch_size = 15
verbose = 1
n_hidden1 = 35
n_hidden2 = 20
n_hidden3 = 15

# Split the dataset into train, validation, and test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# Print train, validation, and test data
print('\nTrain dataset: ', len(X_train))
print('Validation dataset: ', len(X_val))
print('Test dataset: ', len(X_test),)

#Scaling training, validation, and test input data:
mean_vals = np.mean(X_train)
std_val = np.std(X_train)
X_train_std = (X_train - mean_vals)/std_val
X_val_std = (X_val - mean_vals)/std_val
X_test_std = (X_test - mean_vals)/std_val


# print the tranning and testing data:
print("\nTraining Scale: ", X_train_std.shape, y_train.shape)
print("Validation Scale: ", X_val_std.shape, y_val.shape)
print("Testing Scale: ", X_test_std.shape, y_test.shape)


# Build a model using Sequential
model = tf.keras.models.Sequential()

# Add 3 hidden layers to the model
model.add(keras.layers.Dense(n_hidden1, input_shape=(n_features,), kernel_initializer='zero', activation='sigmoid', name='dense_1'))
model.add(keras.layers.Dense(n_hidden2, input_shape=(n_features,), kernel_initializer='zero', activation='sigmoid', name='dense_2'))
model.add(keras.layers.Dense(n_hidden3, input_shape=(n_features,), kernel_initializer='zero', activation='sigmoid', name='dense_3'))

# Print the model summary
model.summary()

# Compile the model
adam_optimizer= keras.optimizers.Adam(learning_rate = 0.1, beta_1 = 0.9)
model.compile(optimizer=adam_optimizer,
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# Print the fit training model:
print("\nFit model on training data")
ann = model.fit(
    X_train_std,
    y_train,
    batch_size=batch_size,
    epochs=Epochs,
    validation_data=(X_val_std, y_val),
)


# Evaluate the model on the test data
print("\nEvaluate on train data")
train_loss, train_accuracy = model.evaluate(X_train_std, y_train)
print("Train Loss: %.2f" % train_loss)
print("Train Accuracy: %.2f" % train_accuracy)

# Evaluate the model on the test data
print("\nEvaluate on test data")
test_loss, test_accuracy = model.evaluate(X_test_std, y_test)
print("Test Loss: %.2f" % test_loss)
print("Test Accuracy: %.2f" % test_accuracy)

# printing out the training and test accuracy values in a text file
with open("Right_Limb_ALL_output.txt", "w") as text_file:
   text_file.write(format("Train Accuracy: %.2f" % train_accuracy))
   text_file.write(format("\nTest Accuracy: %.2f" % test_accuracy))
 
# Plot the traning loss and traning accuracy
hist = ann.history
fig = plt.figure(figsize= (12,5))
ax = fig.add_subplot(1,2,1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

# Save the left training plot
plt.savefig('Right_training_plot.png')

# Plot the traning loss and traning accuracy
hist = ann.history
fig = plt.figure(figsize= (12,5))
ax = fig.add_subplot(1,2,1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Testing loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Testing accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)


#saving the left testing plot
plt.savefig('Right_testing_plot.png')

# Show the graph
plt.show()
