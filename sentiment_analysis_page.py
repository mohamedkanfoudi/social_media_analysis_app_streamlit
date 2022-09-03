import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
import string 
 
from sklearn.feature_extraction.text import TfidfVectorizer

from urllib.error import URLError


import streamlit as st
from wordcloud import WordCloud


import streamlit as st
import pandas as pd
import numpy as np

import pandas as pd 



import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = set(stopwords.words('english'))
def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data["model"]
vectorTF = data["tf_vector"]



def explore(df):
  # DATA
  st.write('Data:')
  st.write(df)
  # SUMMARY
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
  st.write('Summary:')
  #st.write(df_types)
def get_df(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(file)
  return df
def download_file(df, types, new_types, extension):
  for i, col in enumerate(df.columns):
    new_type = types[new_types[i]]
    if new_type:
      try:
        df[col] = df[col].astype(new_type)
      except:
        st.write('Could not convert', col, 'to', new_types[i])


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset
def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"

def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)
def get_feature_vector(train_fit):
            vector = TfidfVectorizer(sublinear_tf=True)
            vector.fit(train_fit)
            return vector







def one_feature(df , feature):
    df = df[['created_timestamp',feature]]
    return df
def index_resample(df):
    df = df.set_index(['created_timestamp'])
    df.index = pd.to_datetime(df.index)
    if not df.index.is_monotonic:
                    df = df.sort_index()
    df = df.resample('D').mean().pad()
    return df

def min_(dataframe_collection) :
    les_min = {}
    for i in dataframe_collection:
        les_min[i] = dataframe_collection[i]['created_timestamp'].min()
    return les_min

def max_(dataframe_collection) :
    les_max = {}
    for i in dataframe_collection:
        les_max[i] = dataframe_collection[i]['created_timestamp'].max()
    return les_max

def minimax(dataframe_collection , les_min, les_max):
    min_max = {}

    min_max[0] = les_min[0]
    min_max[1] = les_max[0]

    for i in les_min :
            if les_min[i] <= min_max[0] :
                min_max[0] = les_min[i]
    for i in les_max :
            if les_max[i] >= min_max[1] :
                min_max[1] = les_max[i]
    return min_max
def add_avant(min_max , les_min  , i , feature):
    df3 = pd.DataFrame([[min_max[0] ,0 ] , [les_min[i] ,0 ]] , columns=['created_timestamp' , feature])
    df3 = df3.set_index(['created_timestamp'])
    df3.index = pd.to_datetime(df3.index )

    df3 = df3.resample('D').mean().pad()

    return df3
def add_apres(min_max , les_max  , i , feature):
    df4 = pd.DataFrame([[les_max[i] ,0 ] , [min_max[1] ,0 ]] , columns=['created_timestamp' , feature])
    df4 = df4.set_index(['created_timestamp'])
    df4.index = pd.to_datetime(df4.index )

    df4 = df4.resample('D').mean().pad()

    return df4

def add_all(df , df3 , df4):
    df5 = df3.append(df)
    df6 = df5.append(df4)
    return df6
def drop_duplicated(df , df3,df4 ):
    df = df.drop(df.index[len(df3)])
    df = df.drop(df.index[len(df)- len(df4)])
    return df



def col_(col , feature , names ):
    col[0]= "created_timestamp"

    for i in range(len(list(names.values()))):
        col[i+1] = feature+"_"+names[i]
    return col


def start_fin(startfin ,start , fin , min_max, names) :
    start[0] = min_max[0]
    fin[0] = min_max[1]

    for i in range(len(list(names.values()))):
        start[i+1]= 0
        fin[i+1] = 0
    startfin[0] = start
    startfin[1] = fin
    return startfin

def add_columns(df,dataframe_collection ,feature ,names):
    for i in range (len(dataframe_collection)):   
        for j in range (len(dataframe_collection[0])):   
            df[feature+'_'+names[i]][j] = dataframe_collection[i][feature][j]
    return df



def indexof(names , name):
    for i in range(len(names)):
     if names[i] == name:
        return i 
def itemsof(dataframe_collection):
    a = ()
    l = list(a)

    for i in (dataframe_collection):
        items = list(dataframe_collection[i].columns)
        for x in items:
            if x not in l :
                l.append(x)

    return tuple(l)

# Create and generate a word cloud image:
def create_wordcloud(topic):


    # Create text
    topic1 = 'law,contract,fees'
    topic2 = 'students,school,exams'
    topic3 = 'money,money11111,money222,money3,money4444444444444,money5,pastor,church,money6,money7,money8,money9,money10,money11,money12,money13,money14,money15,money16,money17,money18,money19,money20'

    
    if topic == 'topic1':
        topic = topic1
    elif topic == 'topic2':
        topic = topic2
    else:
        topic = topic3

    wordcloud = WordCloud().generate(topic)
    return wordcloud

def word(resultat , n ):

    words = ""
    resultat = resultat.reset_index(drop=True)
    for i in range (1,n):
        words = words +','+ resultat['keyword'][i]

    words = resultat['keyword'][0]  + words
    return words
def convert_df(df):
                            return df.to_csv().encode('utf-8')

def show_sentiment_analysis_page(save_data , save_name):
    dataframe_collection = save_data
    names_tables = save_name
    st.title("Sentiment Analysis for comments")
    df = pd.read_csv('facebook_comments.csv')
    i=0
    dataframe_collection[i]= df
    names_tables[i] = 'facebook_comments'

    df = pd.read_csv('instagram_comments.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'instagram_comments'

    df = pd.read_csv('COMMENT_HISTORY.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'linkedin_comments'

    


    if(len(dataframe_collection) != 0 and len(names_tables) !=0):
       
        
        try:

           


                st.write(
                    """
                ### PREDICTION  |  Using : Logistics Regression model 
                """
                """
                # Accuracy score :  78.79%
                """
                )     

                names = st.multiselect(
                "Choose tables", list(names_tables.values()), names_tables[0]
                 )
                dataframe_collection1 = {} 
                names_tables1 = {}
                j=0
                for name in names : 
                    i = indexof(names_tables , name)                    
                    df1 = save_data[i]
                    df =pd.DataFrame(data=df1) 
                    
                    st.write("name : ",names_tables[i])
                    print(df)
                    explore(df)

                    feature = st.selectbox(
                        'Choose text column',
                        list(df.columns ) , key=name)
                    
                    if not feature:
                        st.error("Please select at least one feature.")
                    else :
                        df = df.rename(columns={feature:'text'})
                        df = df[['id' ,'text']]
                        dataframe_collection1[j] = df
                        names_tables1[j] = names_tables[i]                        
                        j= j+1

                        
                        

                        #expericence = st.slider("Years of Experience", 0, 50, 3)
                        
                        
                ok = st.button("Predict" , 
                                key='name')

                        


                

                        

                            
                if ok:
                            
                    for name in names : 
                            i = indexof(names_tables1 , name)                    
                            df1 = dataframe_collection1[i]
                            df =pd.DataFrame(data=df1) 
                            df = df.rename(columns={feature:'text'})
                            st.write("Predicted table   : ",names_tables[i])


           
           
                            # Creating text feature
                            df.text = df['text'].apply(preprocess_tweet_text)
                            
                            #dataset = load_dataset("C:/Users/HPr/Desktop/Stage/twitter_python/training_train-1M6.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
                            #Preprocess data
                            #dataset.text = dataset['text'].apply(preprocess_tweet_text)

                            #tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
                            
                            test_feature = vectorTF.transform(np.array(df.iloc[:, 1]).ravel())
                            
                            prediction = regressor.predict(test_feature)
                            
                            test_result_ds = pd.DataFrame({'id': df.id,'text':df.text ,'prediction':prediction})
                            #test_result = test_result_ds.groupby(['id']).max().reset_index()
                            test_result = test_result_ds
                            test_result.columns = ['id','text', 'predictions']
                            test_result.predictions = test_result['predictions'].apply(int_to_string)
                            explore(test_result)
                            

                            csv = convert_df(test_result)

                            st.download_button(
                                "Press to Download",
                                csv,
                                "file_prediction_result.csv",
                                "text/csv",
                                key='download-csv'
                            )

                            #print("test_result : ")
                            #print(test_result)
                            

                            # Creating text feature
                            # 
                            #       
        except URLError as e:
            st.error(
                """
                **This app requires internet access.**
                Connection error: %s
            """
                % e.reason
            )

    


    


#https://www.youtube.com/watch?v=b-tawVKOJkY
#Build a Complete Social Media Analysis Dashboard with Dash Plotly in Python
#https://www.youtube.com/watch?v=GjTv0NNybbI
#https://github.com/Coding-with-Adam/Dash-by-Plotly/tree/master/Analytic_Web_Apps/Linkedin_Analysis
