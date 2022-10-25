import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT - messages_filepath - string -  disaster messages file path
            categories_filepath - string - disaster categories file path
    OUTPUT - 
            df - DataFrame - pandas dataframe with the data be merged between disaster_messages.csv and disaster_categories.csv
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how ='inner', on = 'id')
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row = row.str.strip().str[:-2]
    category_colnames = list(row)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] =  categories[column].astype(int).astype(bool).astype(int)
        
    # drop the original categories column from `df`
    df.drop(labels='categories', axis = 1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    return df

def clean_data(df):
    '''
    INPUT - df - DataFrame - pandas dataframe with the data be merged between disaster_messages.csv and disaster_categories.csv
    OUTPUT - 
            df_new - DataFrame - pandas dataframe after cleaned
    '''
    # drop duplicates
    df_new = df.drop_duplicates()
    return df_new


def save_data(df, database_filename):
    '''
    INPUT - df - DataFrame - pandas dataframe with the data be merged between disaster_messages.csv and disaster_categories.csv and be cleaned
            database_filename - string - database filename
    '''
    # create engine for sqlite
    engine = create_engine(f'sqlite:///{database_filename}')
    # write data to database
    table_name = database_filename.split('/')[-1].replace('.db', '')
    df.to_sql(table_name, engine, index=False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()