from keras.models import model_from_json
import numpy as np
import os
from keras import Model
import DistanceUtility as ds
from keras import Model
from keras.datasets import mnist

json_file = open('model_digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_digit.h5")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

def create_max_pool_model(layer,model):
    model_refined = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    train_encoding = model_refined.predict(X_train)
    return model_refined, train_encoding

def compute_test_support():
	model_refined, train_encoding = create_max_pool_model(5,loaded_model)
	file1 = open("Output.txt","a") 
	for i in range(len(X_test)):
		support = ds.compute_support(X_test[i], 1000,model_refined,train_encoding)
		print('Index value : '+str(i)+' Actual label'+str(y_test[i]))
		file1.write('Index value : '+str(i)+' Actual label'+str(y_test[i])+'support'+str(support)+'\n')
		print(support)
	file1.close()

compute_test_support()

'''model_refined, train_encoding = create_max_pool_model(5,loaded_model)

def compute_support(name):
	test = ds.read_image('./SudokoPerfectSolution/'+name+'.png')
	print(loaded_model.predict_classes(test.reshape(1, 28, 28, 1)))
	print('Support For :'+name+'.png')
	support = ds.compute_support(test, 100,model_refined,train_encoding)
	print(support)
	distance = ds.compute_distance(ds.read_image('./SudokoPerfectSolution/'+name+'.png'),ds.read_image('./SudokoPerfectSolution/'+'4'+'.png'),loaded_model)
	print(distance)

compute_support('4or91')
compute_support('4or92')'''
