import tensorflow as tf
import numpy as np

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
tf.random.set_seed(1)
np.random.seed(1)

#diffrence perceptron and neuron
#perceptron use linear function and divide 0 or 1 
#but neuron use non-linear function -> differentiable(미분 가능한) function can use back propagation ->to optimize
#make simple node (AND operation)

class node:
    #self parameter = direct class node itself,and affect to itself
    def __init__(self) -> None:
        self.w = tf.Variable(tf.random.normal([2,1]))
        self.b = tf.Variable(tf.random.normal([1, 1]))

    def __call__(self,x) :
        return self.preds(x)
        #__call__  make class callable object (able to call the class like fucntion /ex)  a = node())
        #ex2) node([0,1]) -> by __call__, return self.preds([0,1])
    def preds(self,x):
            # 순전파 (forward propagation)
        out = tf.matmul(x,self.w)   #matmul = matrix multiflying -> multiply input x and weight(가중치) that declared in init
        out = tf.add(out, self.b)   #add = plus bias value
        out = tf.nn.sigmoid(out)    #sigmoid = graph type of activation function 
        #first multiply x and w, second add bias, third send result value into activation funcion, then print out 0 or 1
        return out

    def loss(self,y_pred, y):   
        #손실함수 : calculate difference between correct answer and pred value
        #To reduce difference
        #회귀에서는 평균제곱오차 , 분류에서는 크로스 엔트로피 사용
        return tf.reduce_mean(tf.square(y_pred - y))

    def train (self,inputs,outputs,learning_rate):
        epochs = range(10)
        for epoch in epochs :
            #with ... as ..: normally open file stream and close it when with phrase is ended.
            #with METHOD as N : open method and run it
            with tf.GradientTape() as t:
                #load def loss (using input data and correct answer)
                current_loss = self.loss(self.preds(inputs), outputs)                
                # 역전파 (back propagation)
                dW, db = t.gradient(current_loss, [self.w, self.b])
                #assign sub = return a modified value subtracted to optimize
                self.w.assign_sub(learning_rate * dW)
                self.b.assign_sub(learning_rate * db)

inputs = tf.constant([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
outputs = tf.constant([[0.0], [0.0], [0.0], [1.0]])

node = node()
node.train(inputs,outputs,0.01)
print(node([[0.0,0.0]]).numpy()[0][0])
#numpy()[0][0] -> numpy () make tensor type(node([[0.0,0.0]]) to metrix type and [0][0] just direct data in index [0][0] 
print(node([[0.0,0.0]]))
print(node([[0.0,1.0]]).numpy()[0][0])
print(node([[1.0,0.0]]).numpy()[0][0])
print(node([[1.0,1.0]]).numpy()[0][0])
'''
assert node([[0.0,0.0]]).numpy()[0][0] < 0.5
assert node([[0.0,1.0]]).numpy()[0][0] < 0.5
assert node([[1.0,0.0]]).numpy()[0][0] < 0.5
assert node([[1.0,1.0]]).numpy()[0][0] >= 0.5'''
#assert CONDITION(조건), ERROR MESSEAGE
#print out if condition is False, 
#if condition is True, 





    