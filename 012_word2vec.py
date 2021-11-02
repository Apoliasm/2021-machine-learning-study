import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 항상 같은 결과를 갖기 위해 랜덤 시드 설정
tf.random.set_seed(1)
np.random.seed(1)

from IPython.display import Image
def simple_word2vec():

    corpus = ['king is a strong man', 
            'queen is a wise woman', 
            'boy is a young man',
            'girl is a young woman',
            'prince is a young king',
            'princess is a young queen',
            'man is strong', 
            'woman is pretty',
            'prince is a boy will be king',
            'princess is a girl will be queen']

    #remove word that is useless in training
    def remove_stop_words(corpus):
        stop_words = ['is', 'a', 'will', 'be']
        results = []
        for text in corpus:
            tmp = text.split(' ')
            for stop_word in stop_words:
                if stop_word in tmp:
                    tmp.remove(stop_word)
            results.append(" ".join(tmp))
        
        return results
    corpus = remove_stop_words(corpus)
    '''first : remove useless words in data and make updated list '''
    words = []
    for text in corpus:
        for word in text.split(' '):
            words.append(word)

    words = set(words)
    #set = not allow duplication

    #make dictionary and match it with one-hot-encoding 
    '''second : encode word to vector that has dimension as many as number of words'''
    word2int = {}

    for i,word in enumerate(words):
        word2int[word] = i

    sentences = []
    for sentence in corpus:
        sentences.append(sentence.split())

    WINDOW_SIZE = 2

    data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[ \
                    max(idx - WINDOW_SIZE, 0) : \
                    min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
                    # \ mean : phrase is too long, so i continue phrase in next line .
                if neighbor != word:
                    data.append([word, neighbor])
    #make word input and label
    #ALGORITHM : max(idx - 2,0)&min(idx+2, len(sentense)) -> refer each side of word considering edge of index
    df = pd.DataFrame(data, columns = ['input', 'label'])
    #print(df.head(30))
    '''third make label dictionary'''
    ONE_HOT_DIM = len(words)
    # function to convert numbers to one hot vectors
    def to_one_hot_encoding(data_point_index):
        one_hot_encoding = np.zeros(ONE_HOT_DIM)
        one_hot_encoding[data_point_index] = 1
        return one_hot_encoding

    X = [] # input word
    Y = [] # target word

    for x, y in zip(df['input'], df['label']):
        X.append(to_one_hot_encoding(word2int[ x ]))
        Y.append(to_one_hot_encoding(word2int[ y ]))
    #convert X,Y to np array
    X_train = np.asarray(X)
    Y_train = np.asarray(Y)

    #to visualize set 2 dimension
    encoding_dim = 2

    #input word set vector dimension as len of one hot encoding
    input_word = Input(shape=(ONE_HOT_DIM,))
    # it dosnt use bias to use only weight value
    encoded = Dense(encoding_dim, use_bias=False)(input_word)
    # decode to one hot encoding
    decoded = Dense(ONE_HOT_DIM, activation='softmax')(encoded)

    # set model input to output
    w2v_model = Model(input_word, decoded)
    #set back propagation
    w2v_model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(w2v_model.summary())
    '''Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 16)]              0
    _________________________________________________________________
    dense (Dense)                (None, 2)                 32
    _________________________________________________________________
    dense_1 (Dense)              (None, 16)                48
    =================================================================
    Total params: 80
    Trainable params: 80
    Non-trainable params: 0
    _________________________________________________________________
    '''

    w2v_model.fit(X_train, Y_train,
                    epochs=1000,
                    shuffle=True, verbose=1)
    #to check weight value : second layer([1]) (first layer is input layer), 
    vectors = w2v_model.layers[1].weights[0].numpy().tolist()
    #numpy.tolist -> make matrix to list

    #first make dataframe with data including each weight value
    w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
    w2v_df['word'] = list(words)    
    w2v_df = w2v_df[['word', 'x1', 'x2']]
    print(w2v_df)

    #drawing plot
    fig, ax = plt.subplots()
    #plt.subplots : print multiple graph(여러개의 그래프를 출력) / plt.figure : make default graph = empty

    for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
        ax.annotate(word, (x1,x2 ))
        
    PADDING = 1.0
    x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
    y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
    x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
    y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
    
    plt.xlim(x_axis_min,x_axis_max)
    plt.ylim(y_axis_min,y_axis_max)
    plt.rcParams["figure.figsize"] = (9,9)

    plt.show()
