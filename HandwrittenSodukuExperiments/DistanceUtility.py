import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.datasets import mnist
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

import Inference

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Computes the support based on euclidean distance based on pooling layers
# Return the confidence per digit for top k instances


def compute_support(test, k, model_refined, train_encoding):
    support = {}
    n_neigh = k
    #model_refined, train_encoding = create_max_pool_model(5)
    test_encoding = model_refined.predict(test.reshape(1, 28, 28, 1))
    nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(train_encoding)
    _, indices = nbrs.kneighbors(np.array(test_encoding))
    list_of_support = [0] * 10
    list_y = y_train[indices]
    #print(list_y)
    for i in range(k):
        list_of_support[list_y[0][i]] = list_of_support[list_y[0][i]] + 1
    for i in range(10):
        #print(str(i) + ':' + str(list_of_support[i]) + '%')
        support[i] = (list_of_support[i]/k)*100
    #Sort in order of confidence
    support = {k: v for k, v in sorted(support.items(), key=lambda item: item[1],reverse = True)}
    #remove the keys with 0 support
    support = {k:v for k,v in support.items() if v != 0}
    return support


def read_image(img_name):
    img = cv2.imread(img_name, 0)
    test = img.reshape(1, 784)
    test = test.astype('float32') / 255
    plt.imshow(test.reshape(28, 28), cmap='gray')
    return test

def compute_distance(test1,test2,model):
    model_refined = Model(inputs=model.inputs, outputs=model.layers[5].output)
    test1_encoding = model_refined.predict(test1.reshape(1, 28, 28, 1))
    test2_encoding = model_refined.predict(test2.reshape(1, 28, 28, 1))
    dist = distance.euclidean(test1_encoding,test2_encoding)
    return dist
