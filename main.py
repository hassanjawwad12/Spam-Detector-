import pandas as pd   
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt
#%matplotlib inline 

data = pd.read_table("table.txt", 
                    sep="\t",
                    header=None,
                    names=["label", "messages"]) 
data.head()

def explore_data(df):  
    ''' 
    Function for exploring the data
    '''
    print("Shape of the data:- ", df.shape) 
    print('======================================================')
    print("Non of Null values:- ", df.isnull().sum().sum()) 
    print('======================================================')
    print("Information about the data:- ", df.info())  
    print('======================================================')
    print("Describing the data:- ", df.describe())
    print('======================================================')

explore_data(data)

# seeing the distribution of classes, this will help us to identify which types 

len_ham = len(data["label"][data.label == "ham"])
len_spam = len(data["label"][data.label == "spam"])

arr = np.array([len_ham , len_spam]) 
labels = ['ham', 'spam'] 
print("Total No. Of Ham Cases :- ", len_ham)
print("Total No. Of Spam Cases :- ", len_spam)

plt.pie(arr, labels=labels, explode = [0.2,0.0] , shadow=True) 
plt.show() 

def text_preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will") 
    return x  

data["Preprocessed Text"] = data["messages"].apply(lambda x: text_preprocess(x))
data.head()

print(data['messages'][0]) 
print('================================================')  
print(data['Preprocessed Text'][0]) 

data["label"] = data.label.map({'ham':0, 'spam':1})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['Preprocessed Text'], 
                                                    data['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

from sklearn.feature_extraction.text import CountVectorizer 
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train) 

predictions = naive_bayes.predict(testing_data) 
predictions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions))) 
print('Precision score: ', format(precision_score(y_test, predictions))) 
print('Recall score: ', format(recall_score(y_test, predictions))) 
print('F1 score: ', format(f1_score(y_test, predictions))) 

doc = pd.Series("This is the 2nd time we have tried 2 contact u...")
test = count_vector.transform(doc)  

naive_bayes.predict(test)
