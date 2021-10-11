import tensorflow as tf

tf.random.set_seed(123)
#set random seed to always have same result

T = 1.0
F = 0.0
bias = 1.0
#first set value of true, false, bias
#perceptron only use step function = 0 or 1
#define function that match each of x and y
def get_AND_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    y = [
        [F],
        [F],
        [F],
        [T]
    ]
    
    return X, y
def get_OR_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    y = [
        [F],
        [T],
        [T],
        [T]
    ]
    
    return X, y

def get_XOR_data():
    #exclusive or data = if data is not same -> True
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    y = [
        [F],
        [T],
        [T],
        [F]
    ]
    
    return X, y

#make perceptron class
X,y = get_AND_data()

class perceptron :
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([3,1]))
        #tf.variable : save tensor into memory
        ##To train data, need variables(변수) to update parameters(매개 변수)

    def train(self,X):
        err =1
        epoch,max_epoch = 0,20
        #epoch = cycle that has been trained on all data
        while err > 0.0 and epoch<max_epoch :
            epoch += 1
            self.optimize(X)
            #optimize = train several time and increase accuracy
            err = self.mse(y,self.pred(X)).numpy()
            print('epoch:',epoch,'mse:',err)
    def pred(self,X):
        return self.step(tf.matmul(X,self.W))
    def mse(self,y,y_hat): 
        #mse = mean squared error(평균제곱오차) : (predicted value and real value)^2
        return tf.reduce_mean(tf.square(tf.subtract(y,y_hat)))
    def step(self,x):
        return tf.dtypes.cast(tf.math.greater(x,0),tf.float32)
    def optimize(self, X):
        """
        퍼셉트론은 경사하강법을 사용한 최적화가 불가능합니다.
        매번 학습을 진행할 때마다 가중치를 아래의 룰에 맞게 업데이트합니다.  

        if target == 1 and activation == 0:  
          w_new = w_old + input  

        if target == 0 and activation == 1:  
          w_new = w_old - input  

        위의 두가지 조건은 아래의 코드로 간단히 구현 가능합니다.  
        """
        delta = tf.matmul(X, tf.subtract(y, self.step(tf.matmul(X, self.W))), transpose_a=True)
        self.W.assign(self.W+delta)

