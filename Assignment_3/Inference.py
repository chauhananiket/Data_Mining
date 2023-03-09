import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import *
import pickle

def predict(test_set) :
    filename = '/content/drive/MyDrive/Projects/DMG_Assignment/Assignment_3/'
    # filename = '/content/drive/MyDrive/DMG_Assignment/Assignment_3/'
    data = test_set
    model = pickle.load(open(filename+'kmeans.pkl', 'rb'))
    
    y = np.array(data['target'])
    data.drop(['target'],axis = 1)
    X = np.array(data)  
    y_d = y - 1
    
    X_pca = PCA(n_components=5).fit_transform(X)
    prediction = model.predict(X_pca)
    
    return list(prediction)