import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath:str, categories_filepath:str) -> pd.DataFrame:
    """
    Load the data from the given `.csv` file paths, merge the data into a single dataframe.
    
    Parameters:
    messages_filepath (str): path to the messages csv file
    categories_filepath (str): path to the categories csv file
    
    Returns:
    df (pd.DataFrame): daataframe of merged data from both given files.
    """
    # read csv files from the given path
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the two dfs into one
    df = messages.merge(categories, how='outer', on = ['id'])
    
    return df


def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Cleaned the given merged dataframe of messages, and categories.
    Parameters:
    df (pd.DataFrame): merged dataframe from both (messages, categories) csv
    Returns:
    df (pd.DataFrame): cleaned dataframe, removed duplicates, child alone column
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from 'df'
    df = df.drop('categories', axis = 1)
    # concatenate the original datafram with the new 'categories' dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    # based on the data analysis
    # column child alone has only one sunique value 0
    # removed the child alone column
    df = df.drop('child_alone', axis = 1)
    # in column related there are three unique values [0,1,2] 
    # map two to one
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x) 
    
    return df


def save_data(df:pd.DataFrame, database_filepath:str) -> None:
    """
    Save the given dataframe to the sql lite database
    Parameters:
    df (pd.DataFrame): dataframe to save to sql-lite database
    database_filepath (str) : path to save the df as sql database (.db)
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse_Table', engine, index=False, if_exists = 'replace')
      


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