
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
df=pd.read_csv("spam.csv",encoding='latin1')
df.head()
y=df['v1'].values
x=df['v2'].values
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
ps=PorterStemmer()
ps.stem("called")
stopword=set(stopwords.words('english'))
x=[re.sub('[a-zA-Z]','',doc)for doc in x]
document=[doc.split() for doc in x]
def convert(words):
    curr_word=list()
    for i in words:
        if i.lower() not in stopword:
            update_word=ps.stem(i)
            curr_word.append(update_word.lower())
    return curr_word
document=[convert(doc) for doc in document]
document=["".join(doc) for doc in document]
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
xtrain,xtest,ytrain,ytest=train_test_split(document,y)
cv=CountVectorizer(max_features=1000)
p=cv.fit_transform(xtrain)
q=cv.transform(xtest)
p.todense()
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(p,ytrain)
clf.score(q,ytest)


