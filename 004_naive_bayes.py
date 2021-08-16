import pandas as pd
from sklearn import naive_bayes  #visualize to metrix type
from sklearn.datasets import load_iris      #data import
from sklearn.model_selection import train_test_split    #separate data to testdata and train data
from sklearn.naive_bayes import GaussianNB          #Gaussain naive bayes
from sklearn import metrics                     # To test this data is classificated in correct label
from sklearn.metrics import accuracy_score
# sklearn 모델의 동일한 결과 출력을 위해 선언합니다.
import numpy as np
import matplotlib.pyplot as plt             #to show plot in script type py
def Gaussian_naive_bayes_with_iris(self):
    dataset = load_iris()               # load data
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)# save data dataframe type(2 dimentional metrix)
    df['target'] = dataset.target       # save label(target) at dataframe
    df.target = df.target.map({0:"setosa", 1:"versicolor", 2:"virginica"})  #fit label to numeric index
    #df.head()       #head() = first 5 rows/ tail() = last 5 rows

    setosa_df = df[df.target == "setosa"]
    versicolor_df = df[df.target == "versicolor"]
    virginica_df = df[df.target == "virginica"]

    ax = setosa_df['sepal length (cm)'].plot(kind='hist')
    setosa_df['sepal length (cm)'].plot(kind='kde', 
                                        ax=ax, 
                                        secondary_y=True, 
                                        title="setosa sepal length", 
                                        figsize = (8,4))
    #plt.show()                      # To check length data show Gaussian distribution style

    X_train,X_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2) 
    #x = data(length,type of flower), y = data's label(target -> type to predict(flower type))
    #separate data train and data

    model = GaussianNB()                    #set train data to naive bayes
    model.fit(X_train, y_train)

    expected = y_test                   #test data with y_test whether prediction is correct
    predicted = model.predict(X_test)
                                        
    print(metrics.classification_report(y_test, predicted))

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer         #To import bernoulli naive bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score              #To check accuracy

def bernoilli_naive_bayes (self):
    #FIRST PHASE : INPUT DATA AND PREPROCESS DATA TO TRAIN
    email_list = [
                {'email title': 'free game only today', 'spam': True},
                {'email title': 'cheapest flight deal', 'spam': True},
                {'email title': 'limited time offer only today only today', 'spam': True},
                {'email title': 'today meeting schedule', 'spam': False},
                {'email title': 'your flight schedule attached', 'spam': False},
                {'email title': 'your credit card statement', 'spam': False}
              ]
    df = pd.DataFrame(email_list)           #set data type of metrix with dataframe 
    df['label'] = df['spam'].map({True:1,False:0})      #express TF to binary 0,1 -> new label(0,1) was made 
    df_x=df["email title"]              #set X to input data and set y to predict True or False
    df_y=df["label"]

    cv = CountVectorizer(binary=True)   
    x_traincv=cv.fit_transform(df_x)
    #countvectorizer = make metrix to alter inputed X data to numeric data(how many those word is appeared)
    #found 17 words in input data -> each word set vector[0]~[16] -> each index set in list and express 0 or 1
    encoded_input=x_traincv.toarray()       #fit correctly text data to binary data 
    cv.inverse_transform(encoded_input[0].reshape(-1,1))    #check which elements are included in array[0]
    #unknown error : encoded_input[0] is definitely 1D array, but why that is treated as 2D array?
    #temporary solution : array.reshape(-1,1) to reshape 1D array
    cv.get_feature_names()                  #check all elements of 17 words

    #SECOND PHASE : TRAIN DATA
    bnb = BernoulliNB()               #import Bernoulli naive bayes TO TRAIN DATA
    y_train=df_y.astype('int')        #astype : treat data in pandas as int type
    bnb.fit(x_traincv,y_train)      #TRAIN

    #THIRD PHASE : INPUT TEST DATA
    test_email_list = [
                    {'email title': 'free flight offer', 'spam': True},
                    {'email title': 'hey traveler free flight deal', 'spam': True},
                    {'email title': 'limited free game offer', 'spam': True},
                    {'email title': 'today flight schedule', 'spam': False},
                    {'email title': 'your credit card attached', 'spam': False},
                    {'email title': 'free credit card offer only today', 'spam': False}
                ]
    test_df = pd.DataFrame(test_email_list)
    test_df['label'] = test_df['spam'].map({True:1,False:0})
    test_x=test_df["email title"]
    test_y=test_df["label"]
    x_testcv=cv.transform(test_x)
    #FOURTH PHASE: TEST IT
    predictions=bnb.predict(x_testcv)     
    accuracy_score(test_y, predictions)

import numpy as np
import pandas as pd
#import to use multinomial naive bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
def multinomial_naive_bayes (self):
    #FIRST PHASE INPUT DATA AND PREPROCESS IT
    review_list = [
                    {'movie_review': 'this is great great movie. I will watch again', 'type': 'positive'},
                    {'movie_review': 'I like this movie', 'type': 'positive'},
                    {'movie_review': 'amazing movie in this year', 'type': 'positive'},
                    {'movie_review': 'cool my boyfriend also said the movie is cool', 'type': 'positive'},
                    {'movie_review': 'awesome of the awesome movie ever', 'type': 'positive'},
                    {'movie_review': 'shame I wasted money and time', 'type': 'negative'},
                    {'movie_review': 'regret on this move. I will never never what movie from this director', 'type': 'negative'},
                    {'movie_review': 'I do not like this movie', 'type': 'negative'},
                    {'movie_review': 'I do not like actors in this movie', 'type': 'negative'},
                    {'movie_review': 'boring boring sleeping movie', 'type': 'negative'}
                ]
    df = pd.DataFrame(review_list)
    df['label'] = df['type'].map({"positive":1,"negative":0})#map positive to 0 and negative to 1
    df_x = df['movie_review']                               #essential need : set x,y
    df_y = df['label']

    cv = CountVectorizer()                    #vectorize text data to numeric data(how many those word is appeared)
    x_traincv=cv.fit_transform(df_x)        
    encoded_input=x_traincv.toarray() 
    print(encoded_input[0].ndim)            #check what number of array dimention
    #cv.inverse_transform(encoded_input[0])    #to check which words is included in first movie review
    cv.get_feature_names()                    #to check all of words in inputed data
    #SEOCND PHASE : TRAIN DATA
    mnb = MultinomialNB()
    y_train=df_y.astype('int')
    mnb.fit(x_traincv,y_train)
    #THIRD PHASE : INPUT TEST DATA
    test_feedback_list = [
                    {'movie_review': 'great great great movie ever', 'type': 'positive'},
                    {'movie_review': 'I like this amazing movie', 'type': 'positive'},
                    {'movie_review': 'my boyfriend said great movie ever', 'type': 'positive'},
                    {'movie_review': 'cool cool cool', 'type': 'positive'},
                    {'movie_review': 'awesome boyfriend said cool movie ever', 'type': 'positive'},
                    {'movie_review': 'shame shame shame', 'type': 'negative'},
                    {'movie_review': 'awesome director shame movie boring movie', 'type': 'negative'},
                    {'movie_review': 'do not like this movie', 'type': 'negative'},
                    {'movie_review': 'I do not like this boring movie', 'type': 'negative'},
                    {'movie_review': 'aweful terrible boring movie', 'type': 'negative'}
                ]
    test_df = pd.DataFrame(test_feedback_list)
    test_df['label'] = test_df['type'].map({"positive":1,"negative":0})
    test_x=test_df["movie_review"]
    test_y=test_df["label"]
    #FOURTH PHASE: TEST IT
    x_testcv=cv.transform(test_x)
    predictions=mnb.predict(x_testcv)
    print(accuracy_score(test_y, predictions))