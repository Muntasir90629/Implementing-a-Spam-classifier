# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:06:24 2021

@author:Md.Muntasirul hoque
"""


#importing Dataset


import  pandas as pd

messages=pd.read_csv('smsspamcollection/SMSSpamCollection',
                     sep='\t', names=["labels","message"])

#Data cleaning and preprocessing

messages.head()

import re 

import nltk 

nltk.download('stopwords')


from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

corpus=[]

for i in range(0,len(messages)):
    
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    
    review=review.lower()
    
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    review=' '.join(review)
    
    corpus.append(review)



# Creating the Bag of Words model


from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=2500)

X=cv.fit_transform(corpus).toarray()


y=pd.get_dummies(messages['labels'])

y=y.iloc[:,1].values



# Train Test Split



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


#classification_report, confusion_matrix



from sklearn.metrics import classification_report, confusion_matrix

conf_m = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('report:', report, sep='\n')


