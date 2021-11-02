import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, Embedding
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.    sequence import pad_sequences
import numpy as np
import pandas as pd

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
tf.random.set_seed(1)
np.random.seed(1)

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
tf.random.set_seed(1)
np.random.seed(1)
def tensorflow_rnn_test():
    inputs = Input(shape=(1,2))
    #set input data(input 2 dimentional data)
    lstm_out, hidden_state, cell_state = LSTM(1, return_state=True)(inputs)
    #make lstm model
    model = Model(inputs=inputs, outputs=[lstm_out, hidden_state, cell_state])
    #model class is composed of inputs and outputs

    data = np.array([
        [ [1,0] ]
    ])
    lstm_out, hidden_state, cell_state  = model.predict(data)
    print("lstm_out: ",lstm_out)
    #output cell
    print("hidden_state: ",hidden_state)
    #상태값 cell : finally last output cell's output and last hidden state's output are same.
    print("cell_state: ",cell_state)
    #memory cell : ramain only data to remember
def only_one_RNN_test():
    inputs = Input(shape=(1,2))
    output, state = SimpleRNN(3, return_state=True)(inputs)
    #simpleRNN : no memory cell, simple structure, one RNN, 3 dimensional vector
    model = Model(inputs=inputs, outputs=[output, state])


    data = np.array([[ [1,2] ]])
    # test input
    output, state = model.predict(data)
    print("output: ",output)
    print("state: ",state)
    #return outputs about 3 RNN layer
    #each layer's output and state value are same
    print(model.layers[1].weights[0])
    '''
    array([[-0.73366153,  0.8796015 ,  0.28695   ],
        [-0.14340228, -0.4558388 ,  0.3122064 ]], dtype=float32)>'''
    #about metrix multiple(행렬곱) input data is (1,2) shape of metrix, hidden layer is (1,3) metrix
    #those weights is (2,3) shape metrix
    print(model.layers[1].weights[1])
    '''
    array([[ 0.2532742 , -0.8955574 ,  0.36582667],
        [ 0.8398052 ,  0.0158366 , -0.5426569 ],
        [-0.4801869 , -0.4446641 , -0.7561047 ]], dtype=float32)>'''
    #weights[1] : about input weights about hidden layer : hidden layer is (1,3) shape and 
    print(model.layers[1].weights[2])
    '''<tf.Variable 'simple_rnn/simple_rnn_cell/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>'''
    #layer is composed of 3dimensional vector, bias also has 3 bias


def RNN_classifying():
    #one-hot-encoding : each word make number to increase efficiency(효율)
    #each parts(품사) is defined as index
    # I      [1,0,0,0]
    # work   [0,1,0,0]
    # at     [0,0,1,0]
    # google [0,0,0,1]
    #
    # I work at google =  [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
    # I google at work =  [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]
    data = np.array([
        [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ],
        [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]
    ])
    inputs=Input(shape=(4,4))
    #(keras)make input layer
    output, state = SimpleRNN(3, return_state=True, return_sequences=True)(inputs)
    #units = 3 -> output type is 3 dimentional vector.
    #WHAT DOES DIMENSION OF VECTOR MEAN?? WHY DID IT SET 3 DIMENTION?
    #return_state = print hidden layer's state(상태값) value(made by tanh)
    model = Model(inputs=inputs, outputs=[output, state])
    output, state = model.predict(data)
    #output(출력값)/state(상태값)
    print("I work at google: ",output[0])
    print("I google at work: ",output[1])
    '''output value
    I work at google:  [[-0.5511679   0.63119435  0.23787172]
    [ 0.15383278  0.0125707  -0.43020415]
    [ 0.81342876 -0.06634543  0.58575886]
    [ 0.06317464 -0.627327    0.1022412 ]]
    I google at work:  [[-0.5511679   0.63119435  0.23787172]
    [ 0.43859527  0.5722203  -0.4715734 ]
    [ 0.93529207 -0.28600124  0.4644266 ]
    [-0.3341642  -0.89246404  0.38852602]]  <-SAME'''
    print("I work at google: state: ",state[0])
    print("I google at work: state: ",state[1])
    '''state value 
    I work at google: state:  [ 0.06317464 -0.627327    0.1022412 ]
    I google at work: state:  [-0.3341642  -0.89246404  0.38852602]  <- SAME
    '''
def LSTM_test():
    inputs = Input(shape=(1,2))
    lstm_out, hidden_state, cell_state = LSTM(1, return_state=True)(inputs)
    #cell_state = memory cell = characteristic of LSTM
    model = Model(inputs=inputs, outputs=[lstm_out, hidden_state, cell_state])
    #composing model with input and output

    data = np.array([
        [ [1,0] ]
    ])

    lstm_out, hidden_state, cell_state  = model.predict(data)
    print("lstm_out: ",lstm_out)
    print("hidden_state: ",hidden_state)
    print("cell_state: ",cell_state)

paragraph_dict_list = [
         {'paragraph': 'dishplace is located in sunnyvale downtown there is parking around the area but it can be difficult to find during peak business hours my sisters and i came to this place for dinner on a weekday they were really busy so i highly recommended making reservations unless you have the patience to wait', 'category': 'food'},
         {'paragraph': 'service can be slower during busy hours but our waiter was courteous and help gave some great entree recommendations', 'category': 'food'},
         {'paragraph': 'portions are huge both french toast and their various omelettes are really good their french toast is probably 1.5x more than other brunch places great place to visit if you are hungry and dont want to wait 1 hour for a table', 'category': 'food'},
         {'paragraph': 'we started with apps going the chicken and waffle slides and chicken nachos the sliders were amazing and the nachos were good too maybe by themselves the nachos would have scored better but after those sliders they were up against some tough competition', 'category': 'food'},
         {'paragraph': 'the biscuits and gravy was too salty two people in my group had the gravy and all thought it was too salty my hubby ordered a side of double egg and it was served on two small plates who serves eggs to one person on separate plates we commented on that when it was delivered and even the server laughed and said she doesnt know why the kitchen does that presentation of food is important and they really missed on this one', 'category': 'food'},
         {'paragraph': 'the garlic fries were a great starter (and a happy hour special) the pancakes looked and tasted great and were a fairly generous portion', 'category': 'food'},
         {'paragraph': 'our meal was excellent i had the pasta ai formaggi which was so rich i didnt dare eat it all although i certainly wanted to excellent flavors with a great texture contrast between the soft pasta and the crisp bread crumbs too much sauce for me but a wonderful dish', 'category': 'food'},
         {'paragraph': 'what i enjoy most about palo alto is so many restaurants have dog-friendly seating outside i had bookmarked italico from when they first opened about a 1.5 years ago and was jonesing for some pasta so time to finally knock that bookmark off', 'category': 'food'},
         {'paragraph': 'the drinks came out fairly quickly a good two to three minutes after the orders were taken i expected my iced tea to taste a bit more sweet but this was straight up green tea with ice in it not to complain of course but i was pleasantly surprised', 'category': 'food'},
         {'paragraph': 'despite the not so good burger the service was so slow the restaurant wasnt even half full and they took very long from the moment we got seated to the time we left it was almost 2 hours we thought that it would be quick since we ordered as soon as we sat down my coworkers did seem to enjoy their beef burgers for those who eat beef however i will not be returning it is too expensive and extremely slow service', 'category': 'food'},
    
         {'paragraph': 'the four reigning major champions simona halep caroline wozniacki angelique kerber and defending us open champion sloane stephens could make a case for being the quartet most likely to succeed especially as all but stephens has also enjoyed the no1 ranking within the last 14 months as they prepare for their gruelling new york campaigns they currently hold the top four places in the ranks', 'category': 'sports'},
         {'paragraph': 'the briton was seeded nn7 here last year before a slump in form and confidence took her down to no46 after five first-round losses but there have been signs of a turnaround including a victory over a sub-par serena williams in san jose plus wins against jelena ostapenko and victoria azarenka in montreal. konta pulled out of new haven this week with illness but will hope for good things where she first scored wins in a major before her big breakthroughs to the semis in australia and wimbledon', 'category': 'sports'},
         {'paragraph': 'stephens surged her way back from injury in stunning style to win her first major here last year—and ranked just no83 she has since proved what a big time player she is winning the miami title via four fellow major champions then reaching the final at the french open back on north american hard courts she ran to the final in montreal only just edged out by halep she has also avoided many of the big names in her quarter—except for wild card azarenka as a possible in the third round', 'category': 'sports'},
         {'paragraph': 'when it came to england chances in the world cup it would be fair to say that most fans had never been more pessimistic than they were this year after enduring years of truly dismal performances at major tournaments – culminating in the 2014 event where they failed to win any of their three group games and finished in bottom spot those results led to the resignation of manager roy hodgson', 'category': 'sports'},
         {'paragraph': 'the team that eliminated russia – croatia – also improved enormously during the tournament before it began their odds were 33/1 but they played with real flair and star players like luka modric ivan rakitic and ivan perisic showed their quality on the world stage having displayed their potential by winning all three of their group stage games croatia went on to face difficult tests like the semi-final against england', 'category': 'sports'},
         {'paragraph': 'the perseyside outfit finished in fourth place in the premier league table and without a trophy last term after having reached the champions league final before losing to real madrid', 'category': 'sports'},
         {'paragraph': 'liverpool fc will return to premier league action on saturday lunchtime when they travel to leicester city in the top flight as they look to make it four wins in a row in the league', 'category': 'sports'},
         {'paragraph': 'alisson signed for liverpool fc from as roma this summer and the brazilian goalkeeper has helped the reds to keep three clean sheets in their first three premier league games', 'category': 'sports'},
         {'paragraph': 'but the rankings during that run-in to new york hid some very different undercurrents for murray had struggled with a hip injury since the clay swing and had not played a match since losing his quarter-final at wimbledon and he would pull out of the us open just two days before the tournament began—too late however to promote nederer to the no2 seeding', 'category': 'sports'},
         {'paragraph': 'then came the oh-so-familiar djokovic-nadal no-quarter-given battle for dominance in the third set there were exhilarating rallies with both chasing to the net both retrieving what looked like winning shots nadal more than once pulled off a reverse smash and had his chance to seal the tie-break but it was djokovic serving at 10-9 who dragged one decisive error from nadal for a two-sets lead', 'category': 'sports'}
]
df = pd.DataFrame(paragraph_dict_list)
df = df[['paragraph', 'category']]
#pandas : data analyzing api & pd.Dataframe() is basic tool
'''
df.head() #first 5 data
df.tail() #last 5 data
'''
def get_vocab_size(df):
    #return total numbers of non-duplicated words
    results = set()
    #set()집합 자료형 : dosnt allow duplication(중복 허락 x), modifying(수정 x)
    a = df['paragraph']
    a = a.str
    a = a.lower()
    a = a.str
    a= a.split()
    a = a.apply(results.update)

    #a.split make list keeping pandas type/
    #  a.apply(results.update) append words in results = set(), but data in a.list() become None? 
    # ㄴ> dont related about apply,update, because just i command print(a) (a = a.apply object)
    # apply : affect to all column in each row once/set.update -> add index in set once
    #ㄴ> append words which doesnt allow duplicating to set, and it operates about each row

    '''df['paragraph'].str.lower().str.split().apply(results.update)'''
    return len(results)
vocab_size = get_vocab_size(df)
print(vocab_size)

paragraphs = df['paragraph'].tolist()

#make words list
encoded_paragraphs = [one_hot(paragraph, vocab_size) for paragraph in paragraphs]
#encoding paragraphs to numbers -> import one_hot(return list type)
def get_max_length(df):
    """
    데이터에서 가장 긴 문장의 단어 갯수를 리턴합니다.
    """
    max_length = 0
    for row in df['paragraph']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length
        
max_length = get_max_length(df)
print (max_length)

padded_paragraphs_encoding = pad_sequences(encoded_paragraphs, maxlen=max_length, padding='post')
#get max_length to do zero padding -> make all data in df same length -> fill empty space to 0
print(padded_paragraphs_encoding)

categories = df['category'].tolist()
def category_encode(category):
    #also encode category data to numeric data
    if category == 'food':
        return [1,0]
    else:
        return [0,1]
encoded_category = [category_encode(category) for category in categories]

#make training model
model = Sequential()
model.add(Embedding(vocab_size,5,input_length=max_length))

#Embedding = alter integer value to vector type (this case alters 5-dimensional vector)
model.add(LSTM(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
#back propagation (using optimizer(adam))
train_X = np.array(padded_paragraphs_encoding)
train_Y = np.array(encoded_category)
model.fit(train_X, train_Y,batch_size=10,epochs=50)
#model.fit -> train it
score, acc = model.evaluate(train_X, train_Y, verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)
print(model.summary())
'''
모델에 대한 간략한 요약은 다음과 같습니다.
[문맥 벡터 생성]
입력값은 단어들의 인덱스이며, 그 길이는 항상 91입니다.(zero_padding)
임베딩 레이어는 인덱스를 받아, 5차원 벡터의 임베딩을 출력합니다.
LSTM 셀은 64차원 벡터의 상태값을 출력합니다.

[문맥 벡터를 사용하여 지문의 주제 분류하기]
주제 분류는 두개의 dense layer를 사용합니다.
첫번째 dense layer는 32개의 노드를 가지고 있습니다.
두번째 dense layer는 2개의 노드를 가지고 있으며, 이 2개의 노드는 소프트맥스의 입력값으로 들어갑니다.
소프트맥스는 각 분류값에 해당할 확률을 출력합니다.'''

