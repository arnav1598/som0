# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Untitled form.csv')
X = dataset.iloc[:, 3:].values
y = dataset.iloc[:, 1].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X0 = LabelEncoder()
X[:, 0] = labelencoder_X0.fit_transform(X[:, 0])
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
labelencoder_X3 = LabelEncoder()
X[:, 3] = labelencoder_X3.fit_transform(X[:, 3])
labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])
labelencoder_X5 = LabelEncoder()
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])
labelencoder_X6 = LabelEncoder()
X[:, 6] = labelencoder_X6.fit_transform(X[:, 6])
labelencoder_X7 = LabelEncoder()
X[:, 7] = labelencoder_X7.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 4, 5, 6, 7])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, [0, 2, 4, 6, 8, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23]]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
map_dimx=15
map_dimy=10
from minisom import MiniSom
som = MiniSom(x = map_dimx, y = map_dimy, input_len = 16, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Creating the map
from collections import defaultdict
mappings = defaultdict(list)
for i, x in enumerate(X):
            mappings[som.winner(x)].append(y[i])

#Visualisation of the mapping
fact=2.
plt.figure(figsize=(map_dimx*fact, map_dimy*fact))
for i, x in enumerate(X):
    winnin_position = som.winner(x)
    plt.text(x=winnin_position[0]*fact+fact/2, 
             y=winnin_position[1]*fact+np.random.rand()*(fact-0.1),
             horizontalalignment='center',
             fontsize='large',
             s=y[i],
             color=(winnin_position[0]/map_dimx, 1-(winnin_position[0]/map_dimx), winnin_position[1]/map_dimy, 1))
plt.xlim([0, map_dimx*fact])
plt.ylim([0, map_dimy*fact])
plt.axis('off')
plt.plot()
