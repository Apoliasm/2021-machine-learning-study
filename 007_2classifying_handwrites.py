import tensorflow as tf
import numpy as np
tf.random.set_seed(678)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("before data = ",x_train[0][8])
print(x_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale #it can divide all data in array without using for loop (Cant it do in list? can only in array?)
#data normalization(데이터 정규화) : By dividing 255, it can make 0~1 -> more comfortable to training
print("after data = ",x_train[0][8])
#array.astype = change data type in all data in array
#x has train images composed of pixel, y has answer(0~9)
#each pixel in image is composed of number 0~255, if color is like white, it has 0,else 255


model = Sequential([
    Flatten(input_shape=(28, 28)), # 데이터 차원 변경
    #originally x_train data is 2d tensor, change data dimension to 1D tensor
    # [28][28] to [783] 
    Dense(256, activation='relu'), # 첫번째 히든 레이어 (h1)
    Dense(128, activation='relu'), # 두번째 히든 레이어 (h2)
    Dropout(0.1), # 두번째 히든 레이어(h2)에 드랍아웃(10%) 적용 
    #drop out:randomly dont use train data in training
    #  to avoid overfitting(과대적합) -> aviod that training is too leaned to train data
    Dense(10), # 세번째 히든 레이어 (logit)
    Activation('softmax') # softmax layer
])

print(model.summary())
'''Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 256)               200960
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290
_________________________________________________________________
activation (Activation)      (None, 10)                0
=================================================================
Total params: 235,146
Trainable params: 235,146
Non-trainable params: 0
_________________________________________________________________
None'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
sparse_categorical_crossentropy:
레이블을 원 핫 인코딩으로 자동으로 변경하여 크로스 엔트로피 측정합니다.
"""

callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=False),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)]
#early stopping : check accuracy in each epoch, and select stop time(point that accuracy decrease)
model.fit(x_train, y_train, epochs=300, batch_size=1000, validation_split = 0.1, callbacks=callbacks)
#train it
results = model.evaluate(x_test,  y_test, verbose = 0)
#choose the highest model and test evaluate
print('test loss, test acc:', results)