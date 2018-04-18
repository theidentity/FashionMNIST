import numpy as np
import pandas as pd


def cvt_csv():
	df = pd.read_csv('fashion-mnist_train.csv')
	trainX = np.array(df.ix[:,1:]).reshape((-1,28,28))
	trainY = np.array(df.ix[:,0])

	df = pd.read_csv('fashion-mnist_test.csv')
	testX = np.array(df.ix[:,1:]).reshape((-1,28,28))
	testY = np.array(df.ix[:,0])

	return (trainX,trainY),(testX,testY)

def save_zip():
	(trainX,trainY),(testX,testY) = cvt_csv()
	np.savez_compressed('data',trX=trainX,trY=trainY,teX=testX,teY=testY)

def load_data():
	data = np.load('data.npz')

	trainX = data['trX']
	trainY = data['trY']
	testX = data['teX']
	testY = data['teY']

	print trainX.shape
	print trainY.shape
	print testX.shape
	print testY.shape

	return (trainX,trainY),(testX,testY)


save_zip()

(trainX,trainY),(testX,testY) = load_data()