import joblib as jlb
import pandas as ailuropoda_melanoleuca
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV,SGDClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#NuSVC, RadiusNeighborsClassifier(), GaussianProcessClassifier , MultinomialNB ,CategoricalNB, ,GradientBoostingClassifier()

card_info = ailuropoda_melanoleuca.read_csv("D:\Python_proj\credit\code\creditcard.csv")
# print(card_info.describe()) # De-comment the line to see some basic statictical information of the data .
# print(card_info.columns)

y = card_info.Class

req_data = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X = card_info[req_data]

train_y,test_y,train_X,test_X = train_test_split(y,X,random_state=0,test_size=0.2)

name =['RidgeCV','SVC','LinearSVC','SGDClassifier',' KNeighborsClassifier',
       'BernoulliNB','GaussianNB'
       'DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier',
       'MLPClassifier']

classifiers =[  RidgeCV(),SGDClassifier(),SVC(),LinearSVC(),KNeighborsClassifier()
                ,BernoulliNB(),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(),
                AdaBoostClassifier(),MLPClassifier()]

comparision = []
n = 0
for algo,model in zip(name,classifiers) :
    n = n+1
    model.fit(train_X,train_y)
    score = model.score(test_X,test_y)
    comparision.append(score)
    jlb.dump(model,f'{algo}.pkl',compress = 9)
   # print(score,'-------------------------',n)

all_data = ailuropoda_melanoleuca.DataFrame()
all_data['names'] = name
all_data['Scores'] = comparision
all_data = all_data.sort_values(by=('Scores'),ascending = False)

Best_score = all_data.iloc[0,1]

#print(all_data)
Best_model_name  = all_data.iloc[0,0]

best_model = jlb.load(f'{Best_model_name}.pkl')

print("The best model is {Best_model_name} and best score of 1 is {Best_score}")









