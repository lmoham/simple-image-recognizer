from PIL import Image
from keras.datasets import cifar10
from matplotlib import pyplot as plt

labels = ['airplane', 'automobile', 'bird', 'cat' , 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Store result into two tuples each consisting of image array and corresponding label
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

from keras.utils import np_utils
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train /= 255
new_X_test /= 255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

# Create base model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

#Saving results of training into a file so that there is no need to train again
import h5py
model.save('Trained_model.h5')