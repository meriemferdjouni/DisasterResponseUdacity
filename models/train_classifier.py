import sys
# import libraries

#Load-data Libraries
import pandas as pd
from sqlalchemy import create_engine

#Text Processing libraries
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

#Model libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


#Save the model
import joblib
from joblib import dump, load
import pickle

#Evaluate the model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Improve the model
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from nltk.util import ngrams
from termcolor import colored, cprint


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    targetnames = list(Y.columns)
    return X, Y,  targetnames

def tokenize(text):
    

   #1. Normalize: Convert to lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
   #2. Tokenizing: split text into words
    tokens = word_tokenize(text)
    
   #3. Remove stop words: if a token is a stop word, then remove it
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    #4. Lemmatize and Stemming
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    clean_tokens = []
    
    for i in lemmed_words:
        clean_tokens.append(i)

    return clean_tokens   


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MLPClassifier()))
    ])
    
    parameters = { 
    
    
    'clf__estimator__learning_rate_init':[0.01],
    'clf__estimator__batch_size': [40, 60]
    #'clf__estimator__alpha': [0.001, 0.01]
    }
    
     
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2)
    #cv.fit(X_train, y_train)
    #y_pred = cv.predict(X_test)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Predit using the trained model
    Y_pred = model.predict(X_test)
    #predicted_y_df = pd.DataFrame(Y_pred, columns = targetnames)
    
    for i, c in enumerate(category_names): 
        cprint(c, 'green') 
 
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        

def save_model(model, model_filepath):
        """ Input:
        model : the trained model
        model_filepath : the name of the model/ path of the model where we want to save it
        """
        
        #Save the model (Serialization)
        joblib.dump(model, model_filepath)
        
        #Or you can use: pickle.dump(model, open(model_filepath, 'wb'))
        pass
        
        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()