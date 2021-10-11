import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#to import only class in module 
tf.random.set_seed(678) #to result always same data

#in module function what does mean parameter *args, **kwargs?
# one aster *args can receive parameters as much as i want, *(pname) saved tuple type
# two aster **kwargs can receive parameters as much as i want, additionally it can declare specific value 
#ex *args -> (name1,name2) / **kwargs(name1 = 100, name2 = 200)

#define x,y
x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y = np.array([0.,1.,1.,0.])

#perceptron is composed of 3 layers -> input layer, hidden layer,output layer
# 1.input layer
# tf.keras.models.sequential() can make layer and add hidden layer by using add method
model = Sequential()
# first hidden layer  
# Dense can make hidden layer (set number of layer units and activation function type)
model.add(Dense(units=2,activation='sigmoid',input_dim=2))
# second hidden layer
model.add(Dense(units=1,activation='sigmoid'))
# define loss function and optimize type
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
#keras api make it possible to make perceptron without making each layer

print(model.summary())
'''Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 2)                 6
first layer = input to sigma(multiple each weights and add bias)
param6 mean : 2 nodes(arrows direct input to sigma) which have 2 weights and each node has bias(bias to sigma)(4+2)
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
second layer = first layer's result of activation to second sigma, then result comes through activation func
param3 mean = one node which have 2 weights and that node has bias(2+1)
_________________________________________________________________
None'''

model.fit(x,y,epochs=15000,batch_size=4,verbose=1)
#batch = a set of data to update weights once(한번에) 
#epochs = period that have used all data
#for example : data has 1000 piece, 5 epochs, 4 batch
# ㄴ> it will use 1000 pieces of data on 5(epoch) times, and 4(batch) pieces of data are managed at once.

print(model.predict(x,batch_size=4))
#predict output for input sample

print("first layer = ",model.layers[0].get_weights()[0])
