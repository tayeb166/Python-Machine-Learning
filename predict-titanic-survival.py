import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load the passenger datap
passengers = pd.read_csv('train.csv')


#Update sex
passengers['Sex']=passengers['Sex'].map({'male': 0,'female':1})

#Udpate Age
passengers['Age'].fillna(inplace=True, value=round(passengers['Age'].mean()))

#PClass
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p: 1 if p ==1 else 0)

passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p ==2 else 0)


#Parse desired values
features = passengers [['Sex', 'Age', 'FirstClass', 'SecondClass']]
survived = passengers['Survived']


#Train test split 
train_features, test_features, train_lables, test_labels = train_test_split(features,survived)


#Scale the feature data
scaler = StandardScaler()
train_features=scaler.fit_transform(train_features)
test_features=scaler.transform(test_features)

#Creating and training the model
model = LogisticRegression()
model.fit(train_features,train_lables)


#Model Score
print(model.score(train_features,train_lables))

#score test
print(model.score(test_features,test_labels))
print (model.coef_)

#print(train_features)
#print(test_features)

#Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Tayeb = np.array([0,24.0,0.0,1.0])

#COmbine passenger arrays
sample_passengers = np.array([Jack, Rose, Tayeb])

#scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

#Mak survival predictions
print(model.predict(sample_passengers))