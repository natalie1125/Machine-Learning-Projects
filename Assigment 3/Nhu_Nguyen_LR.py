#************************************************************************************
# Nhu Nguyen
# EE5321 – HW#3
# Filename: Nhu_Nguyen_LR.py
# Due: 10/14/2020
##

#*************************************************************************************

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Reading the dataset, creating dataframe
path = 'C:/Users/angel/Desktop/FALL 2020/EE/Assignment 3/cars.csv'
col_names = ['buying','maint','doors','persons','lug_boot','safety','label']
cars = pd.read_csv(path, header=None, names=col_names)
cars.drop(["doors", "persons", "lug_boot"], axis = 1, inplace = True)
cars.info()

# Convert to numeric values
cars = cars.replace('low',1)
cars = cars.replace('med',2)
cars = cars.replace('high',3)
cars = cars.replace('vhigh',4)
cars = cars.replace('unacc',1)
cars = cars.replace('acc',2)
cars = cars.replace('good',3)
cars = cars.replace('vgood',4)


# Convert the data to numeric values
cars = cars.values

#Separating data & target information
X = cars[:, :3]
y = cars[:,3]
X,y = X.astype(int), y.astype(int)


#Data split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=1, stratify=y)

# Setup the Pipeline
pipe = Pipeline(steps=[('pca', PCA()), ('logistic', LogisticRegression())])

# Create a parameter grid
C_range = [0.001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = {
    'pca__n_components': [1, 3, 5],
    'logistic__C': C_range,
}

# Instantiate the grid
grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=True, scoring= 'accuracy', refit=True)

# Fit the best estimator into the training model:
grid.fit(X_train,y_train)

# Examine the best model
best_score = grid.best_score_
best_parameters = grid.best_params_
best_estimator = grid.best_estimator_
print('Best score: %.3f' % best_score)
print('Best parameters: ', best_parameters)
print('Best Estimator: ' , best_estimator)

# Print the Test Accuracy Score
scores = cross_val_score(grid, X_train, y_train, scoring='accuracy', cv=5)
print('Accuracy Score: %.3f' % np.mean(scores))

# Confusion Matrix
y_pred = grid.predict(X_test)
confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confusion_matrix)

# Generate the classifcation report
print(classification_report(y_test, y_pred))

#printing out the training and test accuracy values to a text file
LR_file = open("C:/Users/angel/Desktop/FALL 2020/EE/Assignment 3/LR_output.txt","w") 
LR_file.write(str('Accuracy Score: %.3f' % np.mean(scores)))
LR_file.close()
