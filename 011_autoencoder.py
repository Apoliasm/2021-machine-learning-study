import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from IPython.display import Image

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
'''tf.random.set_seed(1)
np.random.seed(1)'''

'''
auto-encoder : Unsupervised model(비지도학습 모델) that print out output result that is similar to input data 
there is hidden layer that has lower dimensional vector than input layer 
in those hidden layer makes packed data that is composed of core characteristic
IMAGE -> ENCODER -> PACKED INFORMATINO -> DECODER -> SIMILLAR IMAGE
'''


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# we will use train data for auto encoder training
x_train = x_train.reshape(60000, 784)

# select only 300 test data for visualization
x_test = x_test[:300]
y_test = y_test[:300]
x_test = x_test.reshape(300, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# this is the size of our encoded representations
encoding_dim = 4

#reshape normalized image to 28*28
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input(image to 3dimensioanl vector)
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input(vector(packed core information) to image)
decoded = Dense(784, activation='sigmoid')(encoded)
#Dense is used with sequential.add(dense ...) -> in this case, 


# this model maps an input to its reconstruction
#model method is composed of input and output
autoencoder = Model(input_img, decoded) 
# this model maps an input to its encoded representation

#To check encoded image and decoded image 
#it dont use in autoencoder model
#encoder in auto-encoder
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#back propagation = decrease loss 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print(autoencoder.summary())
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 3)                 2355
_________________________________________________________________
dense_1 (Dense)              (None, 784)               3136
=================================================================
Total params: 5,491
Trainable params: 5,491
Non-trainable params: 0
_________________________________________________________________
None'''
'''
first : alter image to 28,28 array and alter 28,28 array to 3dimensioal vector
second : 
'''

autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8

fig = plt.figure(1)
ax = Axes3D(fig)

xs = encoded_imgs[:, 0]
ys = encoded_imgs[:, 1]
zs = encoded_imgs[:, 2]

color=['red','green','blue','lime','white','pink','aqua','violet','gold','coral']

for x, y, z, label in zip(xs, ys, zs, y_test):
    c = color[int(label)]
    ax.text(x, y, z, label, backgroundcolor=c)
    
ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(ys.min(), ys.max())
ax.set_zlim(zs.min(), zs.max())

print(plt.show())


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
print(plt.show())
