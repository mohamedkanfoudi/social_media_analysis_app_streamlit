import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
import string 


from sklearn.feature_extraction.text import TfidfVectorizer





def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def explore(df):
  # DATA
  st.write('Data:')
  st.write(df)
  # SUMMARY
  """
  df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
  numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                   'bool'])].index.values
  df_types['Count'] = df.count()
  df_types['Unique Values'] = df.nunique()
  df_types['Min'] = df[numerical_cols].min()
  df_types['Max'] = df[numerical_cols].max()
  df_types['Average'] = df[numerical_cols].mean()
  df_types['Median'] = df[numerical_cols].median()
  df_types['St. Dev.'] = df[numerical_cols].std()
  """
  st.write('Summary:')
  #st.write(df_types)


def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

vectorTF = data["tf_vector"]


def convert_df(df):
        return df.to_csv().encode('utf-8')

def show_statement_predict():
    #st.title("Explore data ")

    st.write(
        """
    ### PREDICTION  |  Using : Logistics Regression model 
    """
    """
    # Accuracy score :  78.79%
    """
    )
    st.write("""### We need a statement to predict """)
    user_input = st.text_input("label goes here",value="")

    data = ['0','232999989','Thu Jun 25 07:10:05 PDT 2021','NO_QUERY','INMK69', user_input]
    df2 = pd.DataFrame(
                        [data],
                        columns =['predictions','tweet_id', 'created_at', 'query' , 'user', 'text' ])
    df2.to_csv('df_new.csv')
    df2 = pd.read_csv('df_new.csv')
    try : 
        df2 = remove_unwanted_cols(df2, ["Unnamed: 0","predictions", "created_at","query", "user"])
    except :     
            try : 
                df2 = remove_unwanted_cols(df2, ["predictions", "created_at","query", "user"])
            except : 
                try : 
                    df2 = remove_unwanted_cols(df2, [ "created_at","query", "user"])
                except : 
                    try : 
                        df2 = remove_unwanted_cols(df2, [ "query", "user"])
                    except : 
                        try : 
                            df2 = remove_unwanted_cols(df2, [ "user"])
                        except: 
                            df2 = pd.DataFrame(
                                data=df2.values,
                                columns =["tweet_id", "created_at", "text"])    
            else : 
                df2 = remove_unwanted_cols(df2, [])
    
    # declaring a data frame  with three rowsand three columns
    #data = [['Mallika', 23, 'Student'], [       'Yash', 25, 'Tutor'], ['Abc', 14, 'Clerk']]
    
    # creating a pandas data frame
    #data_frame = pd.DataFrame(data, columns=['Name', 'Age', 'Profession'])
    

    #data=[['predictions','tweet_id','created_at','query','user','text'],["0","232999989","Thu Jun 25 07:10:05 PDT 2021","NO_QUERY","INMK69",user_input]]
    #data.columns = ['predictions','tweet_id','created_at','query','user','text']

    print(df2)
    explore(df2)
    """
    csv3 = convert_df(data)

    st.download_button(
            "Press to Download",
            csv3,
            "statement_prediction.csv",
            "text/csv",
            key='download-csv'
        )
    """

    
    

    # Convert the dictionary into DataFrame

    #df_input=pd.DataFrame(data)
    ok2 = st.button("Predict2")

    if ok2 :
    
       
        #dataset = load_dataset("C:/Users/HPr/Desktop/Stage/twitter_python/training_train-1M6.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
        #Preprocess data
        #dataset.text = dataset['text'].apply(preprocess_tweet_text)

        #tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
        
        test_feature = vectorTF.transform(np.array(df2.iloc[:, 1]).ravel())
        
        prediction = regressor.predict(test_feature)
        
        test_result_ds = pd.DataFrame({'tweet_id': df2.tweet_id,'text':df2.text ,'prediction':prediction})
        test_result = test_result_ds.groupby(['tweet_id']).max().reset_index()
        test_result.columns = ['tweet_id','text', 'predictions']
        test_result.predictions = test_result['predictions'].apply(int_to_string)
        explore(test_result)



        ####  taper statement =>  df => to_csv  =>  read_csv  puis predict
    
    

      

