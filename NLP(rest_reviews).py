# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:35:20 2020

@author: HOME
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3) #removing the extrachars like quots,dots

import re #re-replace
import nltk#natural language toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords #grouping all the words
from nltk.stem.porter  import PorterStemmer
ps=PorterStemmer() #removes ed,es in words


data=[]
for i in range(0,1000):
    review=dataset["Review"][i]
    review=re.sub('[^a-zA-Z]',' ',review) #replacing all the chars other than alphabets
    review=review.lower()
    review=review.split() #to make recognize the machine of stop words in the sentence we split the sentence to list
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    data.append(review)

from sklearn.feature_extraction.text import  CountVectorizer

cv=CountVectorizer(max_features=1565)
x=cv.fit_transform(data).toarray()
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(init="uniform",activation="relu",units=1565)) #input
model.add(Dense(output_dim=200,init="uniform",activation="relu")) #hidden
model.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))#output

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=25)
y=model.predict(xtest)
y=y>0.5
#r=model.predict(cv.transform["very bad"])

text="it was amazinng"
review1=text
review1=re.sub('[^a-zA-Z]',' ',review1)
review1=review1.lower()
review1=review1.split()
review1=[ps.stem(word) for word in review1 if not word in set(stopwords.words('english')) ]
review1=' '.join(review1)

r1=model.predict([[review1]])

