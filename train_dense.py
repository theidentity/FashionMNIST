import keras
from keras.models import Sequential,load_model,save_model
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Reshape
from keras.layers import Conv2D,MaxPool2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam,SGD,rmsprop
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical   
import numpy as np


num_epochs = 50
batch_size = 1024
num_classes = 10
model_prefix = ''
inp_shape = (784,)
img_width,img_height = 28,28

def load_data():
	data = np.load('data/data.npz')
	trainX = data['trX']
	trainY = data['trY']
	testX = data['teX']
	testY = data['teY']

	trainX = trainX.reshape((-1,28,28,1))
	testX = testX.reshape((-1,28,28,1))

	trainY = to_categorical(trainY, num_classes=num_classes)
	testY = to_categorical(testY, num_classes=num_classes)

	return (trainX,trainY),(testX,testY)

def FCN():
	# 89.71
	global model_prefix
	model_prefix = 'FCN'

	model = Sequential()
	model.add(Reshape((784,),input_shape=(img_width,img_height,1)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def FCN_with_Dropout():
	# 89.84
	global model_prefix
	model_prefix = 'FCN_with_Dropout'

	model = Sequential()
	model.add(Reshape((784,),input_shape=(img_width,img_height,1)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def Large_FCN():
	# 90.07
	global model_prefix
	model_prefix = 'Large_FCN'

	model = Sequential()
	model.add(Reshape((784,),input_shape=(img_width,img_height,1)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes, activation='softmax'))

	return model

# model = FCN()
# model = FCN_with_Dropout()
model = Large_FCN()

model.summary()

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(
	loss='categorical_crossentropy',
	optimizer=opt,
	metrics=['accuracy'])

(trainX,trainY),(testX,testY) = load_data()

train_datagen = ImageDataGenerator(
	rescale=1/255.0)

test_datagen = ImageDataGenerator(
	rescale=1/255.0)

train_generator = train_datagen.flow(
	trainX,
	trainY,
	batch_size = batch_size,
	)

validation_generator = test_datagen.flow(
	testX,
	testY,
	batch_size = batch_size,
	)

checkpointer = ModelCheckpoint(filepath='models/'+model_prefix+'_best.h5', verbose=1, save_best_only=True)

hist = model.fit_generator(
	train_generator,
	epochs = num_epochs,
	validation_data=validation_generator,
	callbacks = [checkpointer]
	)

model.save('models/'+model_prefix+'.h5')

from matplotlib import pyplot as plt
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training loss','Validation loss'])
plt.savefig(model_prefix+'.jpg')