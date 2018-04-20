import numpy as np


def load_data():
	data = np.load('data/data.npz')

	trainX = data['trX']
	trainY = data['trY']
	testX = data['teX']
	testY = data['teY']


	trainX = trainX.reshape((-1,28,28,1))
	testX = testX.reshape((-1,28,28,1))

	return (trainX,trainY),(testX,testY)