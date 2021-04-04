import swifter
import numpy as np
import pandas as pd
import seaborn as sns
import re
from csv import writer
import copy
import os 
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import mysql.connector

mydb = mysql.connector.connect(host='73.38.248.152', user='buspark', password='U@5p1r3!')

if (mydb):
    print("Connection Successful")
else:
    print("Connection Unsuccessful")

mycursor = mydb.cursor()


# Load sql to dataframe 
# Get Training Set (Action != NULL and Actor != NULL)
# Getting 20000 values first 
case_index_not_null = pd.read_sql("SELECT * FROM wp_courtdocs.cdocs_case_action_index as c_a_index WHERE c_a_index.action != ' ' and c_a_index.actor != ' ' LIMIT 20000;", con = mydb)
columns = ['actor','action','description']
trainSet = case_index_not_null[columns]
print(trainSet.head())

# Get Test Set (Action = NULL)
# Getting 20000 values first 
action_null = pd.read_sql("SELECT * FROM wp_courtdocs.cdocs_case_action_index as c_a_index WHERE c_a_index.action = ' ' LIMIT 20000;", con = mydb)
testSet = action_null[columns]
print(testSet.head())

# Get Distinct Values of Actions Field with Index Number 
path1 = 'C:\\Users\\Serra\\Desktop\\civera\\distinct-case-actions.csv'
distinct_actions = pd.read_csv(path1)
print(distinct_actions.head())

trainSet = trainSet.merge(distinct_actions, on='action')
print(trainSet.head())

r = re.compile(r'[^\w\s]+')
trainSet['description'] = [r.sub('', x) for x in trainSet['description'].tolist()]
trainSet['description'] = trainSet['description'].str.lower().str.split()
testSet['description'] = [r.sub('', x) for x in testSet['description'].tolist()]
testSet['description'] = testSet['description'].str.lower().str.split()

stopwords = stopwords.words('english')
trainSet['description'] = trainSet['description'].apply(lambda x: [item for item in x if item not in stopwords])
print("stopwords")
print(trainSet.head())
print()
testSet['description'] = testSet['description'].apply(lambda x: [item for item in x if item not in stopwords])
print(testSet.head())


lemmatizer = WordNetLemmatizer() 
trainSet['description'] = trainSet['description'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
testSet['description'] = testSet['description'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])

#remove duplicate words after lemmatizing 
trainSet['description'] = trainSet['description'].apply(lambda x:list(dict.fromkeys(x)))
print()
print('trainingSet after lemmatizer & removing dupes ')
print(trainSet.head())

testSet['description'] = testSet['description'].apply(lambda x:list(dict.fromkeys(x)))
print()
print('testSet after lemmatizer & removing dupes ')
print(testSet.head())

#copy
trainSet1 = copy.deepcopy(trainSet)
testSet1 = copy.deepcopy(testSet)

#join back 
trainSet1['description'] = trainSet1 ['description'].apply(lambda x:' '.join(x))
testSet1['description'] = testSet1['description'].apply(lambda x:' '.join(x))
print()

trainSet1['description'] = trainSet1['description'].astype('str')
testSet1['description'] = testSet1['description'].astype('str')

print("preprocessing done")

X = trainSet1['description']
y = trainSet1['action_index']

print("train-test-split processing")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

#path2 = 'C:\\Users\\Serra\\Desktop\\civera\\action_train.csv'
#path3 = 'C:\\Users\\Serra\\Desktop\\civera\\action_test.csv'
#trainSet1.to_csv(path2, mode='w', index = False, header = False)
#testSet1.to_csv(path3, mode='w', index = False, header = False)


# clf = Pipeline([('tfidf', TfidfVectorizer()),('lsvc', LinearSVC(dual=False,C = 0.2)),])

# # training data through the pipeline
# clf.fit(X_train, y_train)

clf2 = Pipeline([('tfidf', TfidfVectorizer()),('rdf',RandomForestClassifier()),])
# training data through the pipeline
clf2.fit(X_train, y_train)

clf3 = Pipeline([('tfidf', TfidfVectorizer()),('mnb',MultinomialNB()),])
# training data through the pipeline
clf3.fit(X_train, y_train)

# estimators=[('SVC',clf),('RDF',clf2),('MNB',clf3)]
# votingclassfier for all the models
# ensemble = VotingClassifier(estimators, voting='hard')
# #fit model to training data
# ensemble.fit(X_train, y_train)

predictions = clf3.predict(testSet1['description'])
print(predictions.shape)

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'actor':testSet1['actor'],'description':testSet1['description'],'action_index':predictions})
#Visualize the first 5 rows
print("prediction")
print(submission.head())
submission = submission.merge(distinct_actions, on='action_index') 

path4 = 'C:\\Users\\Serra\\Desktop\\civera\\multiNB-prediction.csv'
#path5 = 'C:\\Users\\Serra\\Desktop\\civera\\RandomF-prediction.csv'
submission.to_csv(path4, mode='w', index = False)
#submission.to_csv(path5, mode='w', index = False)
#testSet1.to_csv(path3, mode='w', index = False, header = False)

