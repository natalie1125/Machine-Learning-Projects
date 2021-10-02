
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

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
pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reduce_dims', PCA(n_components=2)),
        ('clf', SVC(kernel = 'linear', C = 1, gamma = 1))])

# Create a parameter grid
C_range = [0.001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
gamma_range = [0.001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = dict(reduce_dims__n_components=[1,4,8],
                  clf__C= C_range,
                  clf__gamma= gamma_range,
                  clf__kernel=['rbf','linear'])

# Instantiate the grid
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=True, scoring= 'accuracy', refit=True)

#Fit the best estimator into the model
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

# Print the classification report
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Print the confusion Matrix
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

# printing out the training and test accuracy values to a text file
SVM_file = open("C:/Users/angel/Desktop/FALL 2020/EE/Assignment 3/SVM_output.txt","w") 
SVM_file.write(str('Accuracy Score: %.3f' % np.mean(scores)))
SVM_file.close()
