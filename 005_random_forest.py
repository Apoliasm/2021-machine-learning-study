
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def randomforest ():
    mnist = datasets.load_digits()                  #input data in sklearn.datasets
    features, labels = mnist.data, mnist.target     


    def cross_validation(classifier,features, labels):          
        cv_scores = []

        for i in range(10):
            scores = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
        
        return cv_scores
    #cross_validation(교차 검증) : validate my input data by using a part of train_data and repeat it with other data
    dt_cv_scores = cross_validation(tree.DecisionTreeClassifier(), features, labels)
    rf_cv_scores = cross_validation(RandomForestClassifier(), features, labels)
    cv_list = {    
                'random_forest':rf_cv_scores,
                'decision_tree':dt_cv_scores,
            }
    df = pd.DataFrame.from_dict(cv_list)    #which model expresses high quality
                                            #can make dataframe by using pd.dataframe.from_dict
    #df.plot  -> dataframe의 메소드엔 plot도 있다.              

from sklearn import datasets                        #to use various model (ensemble)
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# sklearn 모델의 동일한 결과 출력을 위해 선언합니다.
import numpy as np
np.random.seed(5)

mnist = datasets.load_digits()                      #load data
features, labels = mnist.data, mnist.target            #training consist of feature(x) and label(y) (In which label is this feature classified)
X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2)       #split train and test data


dtree = tree.DecisionTreeClassifier(                                #non-ensemble train models 
    criterion="gini", max_depth=8, max_features=32,random_state=35) #which model,ensemble or single, show higher accuracy 

dtree = dtree.fit(X_train, y_train)
dtree_predicted = dtree.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=299).fit(X_train, y_train)
knn_predicted = knn.predict(X_test)

svm = SVC(C=0.1, gamma=0.003,
          probability=True,random_state=35).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)

print("[accuarcy]")
print("d-tree: ",accuracy_score(y_test, dtree_predicted))
print("knn   : ",accuracy_score(y_test, knn_predicted))
print("svm   : ",accuracy_score(y_test, svm_predicted))

svm_proba = svm.predict_proba(X_test)       # ~.predict_probe : to show accuracy of each label
print(svm_proba[0:2])                   #show each accuracy of 2 test data (index 0,1)

#hard_voting : choose the highest accuracy among various single models
voting_clf = VotingClassifier(estimators=[                         
    ('decision_tree', dtree), ('knn', knn), ('svm', svm)], 
    weights=[1,1,1], voting='hard').fit(X_train, y_train)
hard_voting_predicted = voting_clf.predict(X_test)
print("hard voting : %f"%accuracy_score(y_test, hard_voting_predicted))

#soft_voting : show accuracy of each model's return,and sum all of accuracy and choose the highest value
voting_clf = VotingClassifier(estimators=[                         
    ('decision_tree', dtree), ('knn', knn), ('svm', svm)], 
    weights=[1,1,1], voting='soft').fit(X_train, y_train)
soft_voting_predicted = voting_clf.predict(X_test)
accuracy_score(y_test, soft_voting_predicted)
print("soft voting : %f"%accuracy_score(y_test, soft_voting_predicted))

#check visualized result with matplotlib.pyplot
import matplotlib.pyplot as plt         
import numpy as np


x = np.arange(5)
plt.bar(x, height= [accuracy_score(y_test, dtree_predicted),
                    accuracy_score(y_test, knn_predicted),
                    accuracy_score(y_test, svm_predicted),
                    accuracy_score(y_test, hard_voting_predicted),
                    accuracy_score(y_test, soft_voting_predicted)])
plt.xticks(x, ['decision tree','knn','svm','hard voting','soft voting']);

plt.show()