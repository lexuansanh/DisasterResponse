from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def load_data(database_filepath):
    '''
    INPUT - database_filepath - string - database file path contain merge data messages and categories

    OUTPUT - X - Series - pandas series is text input data for model
             y - DataFrame - pandas dataframe with categories_name for classification (one hot format)
             category_names - List -  List of categories of lables
    '''

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')

    table_name = database_filepath.split('/')[-1].replace('.db', '')

    df = pd.read_sql_table(table_name, engine)

    X = df['message']

    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    INPUT - text - string - text need to tokenize

    OUTPUT - stemmed_text - List - List of tokenizes of text
    '''
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize
    tokenized_text = word_tokenize(text)

    # remove stop text
    tokenized_text = [
        w for w in tokenized_text if w not in stopwords.words("english")]

    # Stemming
    stemmed_text = [PorterStemmer().stem(w) for w in tokenized_text]

    return stemmed_text


def build_model():
    '''
    OUTPUT - cv - model - model afted apply grip search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'clf__estimator__n_estimators': [150, 200],
        'clf__estimator__max_depth': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT - model - model - predict model
            X_test - series - input data for testing
            y_test - categories - lables of input data for testing (one hot format)
            category_names - List -  List of categories of lables
    '''
    Y_pred = model.predict(X_test)

    for col_id in range(len(category_names)):
        print(classification_report(
            Y_pred[:, col_id], Y_test[category_names[col_id]]))


def save_model(model, model_filepath):
    '''
    INPUT - model - model - predict model

    OUTPUT - model_filepath - string - file path of model after convert model to plk file
    '''
    # save model with pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
