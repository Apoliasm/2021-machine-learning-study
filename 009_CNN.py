import tensorflow as tf
from tensorflow.python.keras.backend import conv2d, pool2d

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
tf.random.set_seed(1)
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("test data has " + str(x_test.shape[0]) + " samples")
print("every test data is " + str(x_test.shape[1]) 
      + " * " + str(x_test.shape[2]) + " image")
print("test data has " + str(x_train.shape[0]) + " samples")
print("every test data is " + str(x_train.shape[1]) 
      + " * " + str(x_test.shape[2]) + " image")
#can print how many have datas -> tf.shape -> can print out shape of tensor

import numpy as np
#originally tensor 60000,28,28 -> numpy array 60000,28,28,1 
#to put data in input layer, change data type
x_train = np.reshape(x_train, (60000,28,28,1))
x_test = np.reshape(x_test, (10000,28,28,1))

#data normalization(데이터 정규화) -> make 0~255 number data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
#to find cross entropy in loss function


#modelization
model = Sequential()
#first input-conv layer node
model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28,28,1),padding='same'))
    #it has 16 filters -> through that node, make conv2d layer
model.add(MaxPooling2D(pool_size=(2,2)))
#pool size (2,2) : divide tensor into 2 / (28,28,16) -> (14,14,16) 
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',padding='same'))
#32filters with weights value
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#two dense node: first 128 nodes,second 10 nodes
#128 nodes take numeric data about characteristic that came through many filters in conv layer and calculate weights
#10 nodes classify which number it is,(0~9) To do it, calculate each number's property. 
#calculate differnce each node value and genuine value by using cross entropy

#first input image
#second cross filter and calculate with filter's weight value and input value
#third make simply with maxpooling, then extract image's characteristic
#fourth multiple perceptron type and calculate each node's property, and clssify which number is
#final back propagation, adjust weights and increase accuracy
print(model.summary())

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=False),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)]
#if dont have improving, stop training 

model.fit(x_train, y_train,
          batch_size=500,
          epochs=5,
          verbose=1,
          validation_split = 0.1, 
          callbacks=callbacks)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


