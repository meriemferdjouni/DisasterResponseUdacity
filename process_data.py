import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    Output:
    
        df: Merged Dataset (Pandas DataFrame)
    """
    #read in CSV files as Pandas Dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #passing inner to how parameter keeps only rows where the 'id' exist
    df = pd.merge(messages, categories, how='inner', on='id') 
    
    return df

def clean_data(df):
    """
    Input:
       df: merged df of messages and categories from previous function
    Output:
        df: clean df
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0] # select the first row of the categories dataframe
    categories.columns  = list(map(lambda i: i[ : -2], row)) 
     
    
    #Convert category values to just numbers 0 or 1
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x : x[-1] )
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
   
    #Replace categories column in df with new category columns
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
   
     #Remove duplicates
    df = df.drop_duplicates()
    #Remove Null values
    df = df.dropna() 
    #Drop the rows that have value 2
    #df.drop(df[df['related'] == 2].index, inplace = True)   
    
    return df
    
    
def save_data(df, database_filename):
    """
    Input:
        Clean pandas df
        database_filename, database file(.db) destination path
    Output:
        saved SQLite database
    """
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)
    pass


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