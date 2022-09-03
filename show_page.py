import streamlit as st
import pandas as pd
import snowflake
import snowflake.connector
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

data = {}
name = {}

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
  st.write(df_types)
def download_file(df, types, new_types, extension):
  for i, col in enumerate(df.columns):
    new_type = types[new_types[i]]
    if new_type:
      try:
        df[col] = df[col].astype(new_type)
      except:
        st.write('Could not convert', col, 'to', new_types[i])
def transform(df):
  frac = st.slider('Random sample (%)', 1, 100, 100)
  if frac < 100:
    df = df.sample(frac=frac/100)
  
  cols = st.multiselect('Columns'
                        ,df.columns.tolist()
                        ,df.columns.tolist())
  df = df[cols]
  types = {'-':None
           ,'Boolean': '?'
           ,'Byte': 'b'
           ,'Integer':'i'
           ,'Floating point': 'f' 
           ,'Date Time': 'M'
           ,'Time': 'm'
           ,'Unicode String':'U'
           ,'Object': 'O'}
  new_types = {}
  expander_types = st.beta_expander('Convert Data Types')
  for i, col in enumerate(df.columns):
    txt = 'Convert {} from {} to:'.format(col, df[col].dtypes)
    expander_types.markdown(txt, unsafe_allow_html=True)
    new_types[i] = expander_types.selectbox('Field to be converted:'
                                            ,[*types]
                                            ,index=0
                                            ,key=i)
  st.text(" \n") #break line
  # first col 15% the size of the second  
  col1, col2 = st.beta_columns([.15, 1])
  with col1:
    btn1 = st.button('Get CSV')
  with col2:
    btn2 = st.button('Get Pickle')
  if btn1:
    download_file(df, types, new_types, "csv")
  if btn2:
    download_file(df, types, new_types, "pickle")
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

def show_show_page(dataframe_collection, names_tables):
    st.title('Download and Show a dataset')
    st.write('this app can connect to your Snowflake Cloud to get a table as dataframe')
    #st.write('as you can upload a csv or xlsx file from your computer as dataframe')


    st.write("""###Enter your connection informations of Snowflake""")
        
    username = st.text_input("username : ",value="MOHAMEDELKANFOUDI")
    password = st.text_input("password :",value="" , type='password')
    account_name = st.text_input("account_name : (https://<account_name>.snowflakecomputing.com ) ",value="ao28328.central-us.azure")
    warehouse = st.text_input("warehouse : ",value="PC_FIVETRAN_WH")
    database = st.text_input("database : ",value="PC_FIVETRAN_DB")
    schema = st.text_input("schema",value="KERTYS_SOCIAL_MEDIA_REPORTING")
    role = st.text_input("role : ",value="ACCOUNTADMIN")
    table = st.text_input("table1  : ",value="SOCIAL_MEDIA_REPORTING__FACEBOOK_POSTS_REPORTING")

    table1 = st.text_input("table2  : ",value="SOCIAL_MEDIA_REPORTING__INSTAGRAM_POSTS_REPORTING")
    table2 = st.text_input("table3  : ",value="SOCIAL_MEDIA_REPORTING__LINKEDIN_POSTS_REPORTING")
    table3 = st.text_input("table4  : ",value="")

    
    i=0

  
    ok10 = st.button("1/ connect to Snowflake ")
    if ok10 :
            ok10 = False


            url = URL(
                user=username,
                password=password,
                account=account_name,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role = role
            )
            engine = create_engine(url)
            
            
            connection = engine.connect()
            st.write('the connection to Scnowflake is successful !')

            query = '''
                select * from ''' + table
            query1 = '''
                select * from ''' + table1
            query2 = '''
                select * from ''' + table2
            query3 = '''
                select * from ''' + table3



            df = pd.read_sql(query, connection)
            #st.write(df.head(10))
            df.to_csv(table+'.csv')
            #df2 = pd.read_csv('df_new.csv')



        



            names_tables[i] = table+''        
            dataframe_collection[i] = pd.DataFrame(df, columns=df.columns)


            if(table1  is not "") :
                df1 = pd.read_sql(query1, connection)

                df1.to_csv(table1+'.csv')
                #df2 = pd.read_csv('df_new.csv')
                i=i+1
                names_tables[i] = table1+''        
                dataframe_collection[i] = pd.DataFrame(df1, columns=df1.columns)


            if(table2 is not "") :
                df2 = pd.read_sql(query2, connection)

                df2.to_csv(table2+'.csv')
                #df2 = pd.read_csv('df_new.csv')
                i=i+1
                names_tables[i] = table2+''        

                dataframe_collection[i] = pd.DataFrame(df2, columns=df2.columns)

            if(table3 is not "") :
                df3 = pd.read_sql(query3, connection)

                df3.to_csv(table3+'.csv')
                #df2 = pd.read_csv('df_new.csv')
                i=i+1
                names_tables[i] = table3+''        
                dataframe_collection[i] = pd.DataFrame(df3, columns=df3.columns)


          
           
            if(len(dataframe_collection) != 0  and len(names_tables)!=0):
                for j in range(len(dataframe_collection)):
                            #tables = st.multiselect(
                             #   "Choose tables", list(names_tables2['name']) , list(names_tables2['name'])[0] 
                             #)
                            #for j in tables :    
                              st.write(j)                
                              st.write('file name : ' , names_tables[j] )
                              st.write(dataframe_collection[j])       
                              #for i in range (len(names_tables2)):
                               #   if names_tables2.iloc[i][1] == j :
                                #    print(i)

#                            j= j+1

    data = dataframe_collection
    name = names_tables

def save_data():
    return data 
def save_name():
    return name     





#https://docs.streamlit.io/library/api-reference/charts/st.line_chart
#https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
#show_show_page(data)