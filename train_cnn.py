import keras
from keras.models import Sequential,load_model,save_model
from keras.layers import Dense,Activation,Flatten,Dropout
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
img_width,img_height = 28,28
model_prefix = ''
inp_shape = (img_width,img_height,1)


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

def CNN_bare():
	# 89.27
	global model_prefix
	model_prefix = 'CNN_bare'

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inp_shape))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def CNN():
	# 89.10
	global model_prefix
	model_prefix = 'CNN'

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inp_shape))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def CNN_with_BN():
	# 92.43
	global model_prefix
	model_prefix = 'CNNBn'

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inp_shape))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(num_classes, activation='softmax'))

	return model

# model = CNN_bare()
# model = CNN()
model = CNN_with_BN()
# model = getModel2()
# model = Xception()
model.summary()

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# opt = rmsprop(lr=0.0001, decay=1e-6)
# opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

model.compile(
	loss='categorical_crossentropy',
	optimizer=opt,
	metrics=['accuracy'])

(trainX,trainY),(testX,testY) = load_data()

train_datagen = ImageDataGenerator(
	# rotation_range=0.2,
	# zoom_range=0.2,
	# width_shift_range=0.2,
	# height_shift_range=0.2,
	rescale=1/255.0,
	# horizontal_flip=True,
	fill_mode='nearest')

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

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
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