
# Importing all required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Downloading the dataset, creating dataframe
car_dataset = pd.read_csv('C:/Users/angel/Desktop/FALL 2020/EE/Assignment 3/cars.csv')
col_names = ['buying','maint','doors','persons','lug_boot','safety','label']
car_dataset.drop(["maint", "doors", "persons", "lug_boot"], axis = 1, inplace = True)
car_dataset.info()

# Convert to numeric values
car_dataset = car_dataset.replace('low',1)
car_dataset = car_dataset.replace('med',2)
car_dataset = car_dataset.replace('high',3)
car_dataset = car_dataset.replace('vhigh',4)
car_dataset = car_dataset.replace('unacc',1)
car_dataset = car_dataset.replace('acc',2)
car_dataset = car_dataset.replace('good',3)
car_dataset = car_dataset.replace('vgood',4)

# Convert pandas to numpy
cars = car_dataset.values

#Separating data & target information
X = cars[:, :2]
y = cars[:,2]
X,y = X.astype(int), y.astype(int)

#Data split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
random_state=1, stratify=y)

#Scaling training data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Creating perceptron with hyperparameters
ppn = Perceptron(max_iter=40, eta0=0.45, random_state=1)

#This is training the model
ppn.fit(X_train_std, y_train)


#Scaling test data
sc.fit(X_test)
X_test_std = sc.transform(X_test)

#Testing the model data
y_pred = ppn.predict(X_test_std)

# View the predict test data
y_pred

# View model accuracy
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

# Visualize 
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

# Plot of the decision regions
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('High Safety')
plt.ylabel('>= High Prices')
plt.legend(loc='upper left')
plt.show()

