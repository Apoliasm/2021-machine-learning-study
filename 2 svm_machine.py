import pandas as pd                  #High-performing machine learning model is a model with few errors in testing
import pickle               #save data of non-txt data
import matplotlib.pyplot as plt                
import seaborn as sns
'''
with open("D:/머신러닝학습/basketball_train.pkl", "rb") as train_data:   #open learn-data and test-data with binary-write mode 
    train = pickle.load(train_data)

with open("D:/머신러닝학습/basketball_test.pkl", "rb") as test_data:
    test=  pickle.load(test_data)'''
train = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_test.csv")

from sklearn.model_selection import GridSearchCV        
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np                      #numpy = help with processing metrix

def svc_param_selection(X, y, nfolds):                  #find best parameters(Cost,gamma) by using import GridsearchCV.
    svm_parameters = [
                        {'kernel': ['rbf'],
                         'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1],
                         'C': [0.01, 0.1, 1, 10, 100, 1000]
                        }
                       ]
    
    clf = GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(X_train, y_train.values.ravel())
    
    
    return clf
X_train=train[['3P','BLK']]

y_train=train[['Pos']]

clf = svc_param_selection(X_train, y_train.values.ravel(), 10)  # find best of cost and gamma by def made on upsise line 
print(clf)

C_canditates = []                                           #For checking if cost and gamma which are resulted by def is correct 
C_canditates.append(clf.best_params_['C'] * 0.01)
C_canditates.append(clf.best_params_['C'])
C_canditates.append(clf.best_params_['C'] * 100)
gamma_candidates = []
gamma_candidates.append(clf.best_params_['gamma'] * 0.01)
gamma_candidates.append(clf.best_params_['gamma'])
gamma_candidates.append(clf.best_params_['gamma'] * 100)

X = train[['3P', 'BLK']]
Y = train['Pos'].tolist()                   #tolist() = change np.array type to list type

position = []                               #make vector that seperates position 
for gt in Y:
    if gt == 'C':
        position.append(0)
    else:
        position.append(1)

classifiers = []                        #save data for following each C and gamma in list 
for C in C_canditates:
    for gamma in gamma_candidates:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X, Y)
        classifiers.append((C, gamma, clf))

plt.figure(figsize=(18, 18))                    #make 9 plots
xx, yy = np.meshgrid(np.linspace(0, 4, 100), np.linspace(0, 4, 100))

for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    
    plt.subplot(len(C_canditates), len(gamma_candidates), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu,shading='auto')                #visualize boundary vector, color divide red and blue
    plt.scatter(X['3P'], X['BLK'], c=position, cmap=plt.cm.RdBu_r, edgecolors='k') #scatter x_train data in plot by dots
#plt.show()    check whether SV is correct point

X_test = test[['3P', 'BLK']]        #test phase: Using test data, test train data
y_test = test[['Pos']]


y_true, y_pred = y_test, clf.predict(X_test)  
#By injecting test data into SVM completed with best parameters, get actual and predicted value


print(classification_report(y_true, y_pred))
print()
print("accuracy : "+ str(accuracy_score(y_true, y_pred)) )
#four data result which can express metrix of True and False.
#True positive = predict that data is true and actually is.
#false positive = predict that data is false and actually is.
#True negative = predict that data is True but actually is False.
#True negative = predict that data is False but actually is True.
#four data accuracy 
#accuracy = How many data can predict correctly , (Tpositive+Fpositive) /entire data
#recall (재현률) = How many "True" data can predict correctly , Tpositive/(Tpositive+Fnegative)
#precision(정밀도) = how many data predict True data is actually True, Tpositive/(Tpositive+Fpositive)
#f1_score = harmonic mean(조화평균) of recall and precision, 2*precision*recall / (precision+recall)

comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_true.values.ravel()}) 
print(comparison) #check each predicted result and true data