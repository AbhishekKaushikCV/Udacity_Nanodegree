import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load the data from the database as dataframe.Divide the dataframe into features,and targets.
    Parameters:
    database_filepath : path of the stored database
    Returns:
    X: features of the dataset for training
    Y: targets of the dataset for training
    categories : column names
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse_Table', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize the given text.
    Parameters:
    text : input txt to tokenize
    Returns:
    clean_tokens : tokenized, and lemmatized tokens"""
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # lemmatize
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens 


def build_model():
    """
    Builds Ada Boost Classifier and tunes model parameters (n_estimators, learning rate) using GridSearchCV.
    
    Returns:
    model : Tuned model 
    """    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
        
    parameters = {
        'clf__estimator__n_estimators' : [10, 20],
        'clf__estimator__learning_rate':[0.5, 1.0],
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names) -> None:
    """Evaluate the model's performance, and print the metrics(precision, recall, f1-score).
    Parameters:
    model : model to test the performance
    X_test: test messages
    Y_test: categories for test messages
    category_names: labels of the target 
    Returns:
    None: prints the model's performance
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))


def save_model(model, model_filepath:str) -> None:
    """Save the trained model to a pickle file at the given file path.
    Parameters: 
    model : trained model to save
    model_filepath : path to save the trained model as .pkl file
    """
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


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