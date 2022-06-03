import numpy as np
import pandas as pd
from skimage.io import imread
from matplotlib import pyplot as plt
from galaxyteam.dataset import resize_image
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def read_data(train_info_file_path='./data/preprocessed/train_metadata.csv',test_info_file_path='./data/preprocessed/test_metadata.csv'):
    full_train = pd.read_csv(train_info_file_path)
    train_data, val_data = train_test_split(full_train, 
    test_size=0.2,
    shuffle=True,
    stratify=full_train.is_pneumonia,
    random_state=9473)

    
    test_data = pd.read_csv(test_info_file_path)
    X_train = [imread(path,as_gray=True) for path in train_data.resized_file_path]
    Y_train = np.array([pnu for pnu in train_data.is_pneumonia])
    X_val = [imread(path,as_gray=True) for path in val_data.resized_file_path]
    Y_val = np.array([pnu for pnu in val_data.is_pneumonia])
    X_test = [imread(path,as_gray=True) for path in test_data.resized_file_path]
    Y_test = np.array([pnu for pnu in test_data.is_pneumonia])

    X_train = np.reshape(X_train, (np.shape(X_train)[0], -1))
    X_test = np.reshape(X_test, (np.shape(X_test)[0], -1))
    X_val = np.reshape(X_val, (np.shape(X_val)[0], -1))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test





def get_best_k(X_train, Y_train, X_val, Y_val):
    k_n = []
    F1_n = []
    best_k=0
    f1=0

    for k in range(1, 50):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train,Y_train)
        Y_val_pred = model.predict(X_val) 
        cm = confusion_matrix(Y_val, Y_val_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp)*100
        recall = tp/(tp+fn)*100
        F1_n.append(2*precision*recall/(precision+recall))
        if F1_n[k-1]>f1:
            f1=F1_n[k-1]
            best_k=k-1
        k_n.append(k)

    return best_k





