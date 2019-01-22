import h5py
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, RepeatVector
from keras.callbacks import ModelCheckpoint
import numpy as np

import matplotlib.pyplot as plt

from common import show_test_signals

import time


from_file = True

fitNN = False

if (from_file):

	print('load test data...')

	filenameTest = 'TestDataModulation.hdf5'

	fx = h5py.File(filenameTest, 'r')

	XTest = fx['X']
	YTest = fx['Y']
	Signals = fx['S']
	PRI_HIST = fx['PRI_HIST']

	XTest = np.array(XTest)

	XTest = XTest.reshape(XTest.shape[0], XTest.shape[1], 1)

	YTest = np.array(YTest)

	YTest = YTest.reshape(YTest.shape[0], 2 * YTest.shape[2])

	in_layer = Input((XTest.shape[1], 1))
	x = Conv1D(filters=32, kernel_size=5, kernel_initializer='orthogonal')(in_layer)
	x = MaxPooling1D()(x)
	x = Flatten()(x)
	out = Dense(YTest.shape[1], activation='sigmoid')(x)

	model = Model(inputs=in_layer, outputs=out)

	print('load weights...')

	model.load_weights('weights_conv_modul.hdf5')

	print('Compile model...')

	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	if (fitNN):
		print('load train data...')

		filename = 'TrainDataModul.hdf5'

		f = h5py.File(filename, 'r')

		X = f['X']
		Y = f['Y']

		X = np.array(X)

		X = X.reshape(X.shape[0], X.shape[1], 1)

		Y = np.array(Y)

		Y = Y.reshape(Y.shape[0], 2 * Y.shape[2])

		callback = ModelCheckpoint(filepath='weights_conv_modul.hdf5', monitor='acc', save_weights_only = True)

		history = model.fit(X, Y, epochs=30, batch_size=256, shuffle='batch', callbacks=[callback])

	start_time = time.time()

	perform = model.evaluate(XTest, YTest)

	end_time = time.time()

	print(perform)

	index = 105

	target = YTest[index]

	true_pris = []

	for i in range(target.shape[0]):
		if target[i] > 0:
			k = 0
			t = i
			if (i > 80):
				k = 1
				t = i - 81
			true_pris.append([PRI_HIST[t], k])

	predicted = model.predict(XTest[index : index + 1])


	predicted_pris = []

	for i in range(predicted[0].shape[0]):
		if (predicted[0][i] > 0.05):
			k = 0
			t = i
			if (i > 80):
				k = 1
				t = i - 81
			predicted_pris.append([PRI_HIST[t], predicted[0][i], k])

	print('true:')
	print(true_pris)

	print('predicted:')
	print(predicted_pris)

	print('time of working (sec):')
	print(end_time - start_time)

	show_test_signals(Signals[index], 1 * 10 ** (-6))

	plt.plot(target, 'b', predicted[0], 'g')

	plt.ylabel('Probability')
	plt.xlabel('HIST Bin')

	plt.show()

else:

	print('load train data...')

	filename = 'TrainDataModul.hdf5'

	f = h5py.File(filename, 'r')

	X = f['X']
	Y = f['Y']

	X = np.array(X)

	X = X.reshape(X.shape[0], X.shape[1], 1)

	Y = np.array(Y)

	Y = Y.reshape(Y.shape[0], 2 * Y.shape[2])

	in_layer = Input((X.shape[1], 1))
	x = Conv1D(filters=32, kernel_size=5, kernel_initializer='orthogonal')(in_layer)
	x = MaxPooling1D()(x)
	x = Flatten()(x)
	out = Dense(Y.shape[1], activation='sigmoid')(x)


	model = Model(inputs=in_layer, outputs=out)

	print('Compile model...')
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	model.summary()

	print('Training the model...')

	callback = ModelCheckpoint(filepath='weights_conv_modul.hdf5', monitor='acc', save_weights_only = True)

	history = model.fit(X, Y, epochs=30, batch_size=256, shuffle='batch', callbacks=[callback])

	print('Training is done!')
