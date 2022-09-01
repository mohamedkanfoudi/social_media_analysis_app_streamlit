from argparse import Namespace
from operator import le
from unicodedata import name
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
import string 
import plotly.figure_factory as ff
 
from sklearn.feature_extraction.text import TfidfVectorizer

from urllib.error import URLError


import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import pydeck as pdk
import math

import datetime
import leafmap.foliumap as leafmap


import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import datetime
import time
import pandas as pd 
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

def show_analyse_page(save_data , save_name):
    dataframe_collection = save_data
    names_tables = save_name
    st.title("Analyse")
    df = pd.read_csv('FACEBOOK.csv')
    i=0
    dataframe_collection[i]= df
    names_tables[i] = 'FACEBOOK'

    df = pd.read_csv('INSTAGRAM.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'INSTAGRAM'

    df = pd.read_csv('LINKEDIN.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'LINKEDIN'

    


    if(len(dataframe_collection) != 0 and len(names_tables) !=0):
       
        
        try:

           


            ''''''''''''''''''''
            st.write("### MAP  ")

            df = pd.read_csv('KEYWORD_PAGE_REPORT.csv')
            df = df.groupby(by='country', as_index=False).agg({'impressions': 'sum' , 'clicks':'sum' })
            df = df.sort_values(by = 'clicks' , ascending=False  )
            df = df.reset_index(drop=True)
            df['lat'] = 0
            df['lng'] = 0
            df1 = pd.read_csv('countries.csv')

            for j in range(len(df1)):
                for i in range(len(df)):
                    if (df['country'][i].lower() == df1['iso3'][j].lower() and df['lat'][i]==0 and df['lng'][i]==0 ):
                        df['lat'][i] = df1['lat'][j]
                        df['lng'][i] = df1['lon'][j]
                     
            df =df.rename(columns={"lng": "lon"})

            ### SELECTBOX widgets
            metrics =['impressions','clicks']

            cols = st.selectbox('metric to view', metrics)
            df = df[df.clicks != 0]
            df = df[df.impressions != 0]
            
            # let's ask the user which column should be used as Index
            if cols in metrics:   
                metric_to_show_in_Layer = cols  

            #SCATTERPLOT_LAYER_DATA = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/bart-stations.json"
            #df = pd.read_json(SCATTERPLOT_LAYER_DATA)

            # Use pandas to calculate additional data
            #df["exits_radius"] = df["exits"].apply(lambda exits_count: math.sqrt(exits_count))

            # Define a layer to display on a map
            layer = pdk.Layer(
                "ScatterplotLayer",
                df,
                pickable=True,
                opacity=0.3,
                stroked=True,
                filled=True,
                radius_scale=20,
                radius_min_pixels=5,
                radius_max_pixels=60,
                line_width_min_pixels=1,
                get_position=["lon", "lat"],
                get_radius=metric_to_show_in_Layer,
                get_fill_color=[255, 140, 0],
                get_line_color=[0, 0, 0],
            )

            # Set the viewport location
            view_state = pdk.ViewState(latitude=32, longitude=-5, zoom=3, bearing=0, pitch=0)

            # Render
            if(metric_to_show_in_Layer=='clicks'):
                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{country}\n({lon},{lat})\n{clicks}"})
            else :
                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{country}\n({lon},{lat})\n{impressions}"})



            # Create a subheading to display current date
            subheading = st.subheader("")

            # Render the deck.gl map in the Streamlit app as a Pydeck chart 
            map = st.pydeck_chart(r)
            ''''''''''''''''''''

 
            

            













            ''''''''''''''''''''
            
            #st.map(df)

            st.write("### PAGE SEARCH ")
            df = pd.read_csv('KEYWORD_PAGE_REPORT.csv')
            df = df[['date','keyword' ,'page', 'clicks' , 'impressions','position']]
            df = df.groupby(by='page', as_index=False).agg({'impressions': 'sum' , 'clicks':'sum' })
            options = st.multiselect('Choose features',['impressions', 'clicks'],['clicks'])

            if not options:
                    st.error("Please select at least one feature.")
            else:
                df = df.sort_values(by = options[0] , ascending=False  )
                st.write("sorted by  " , options[0] )
                df = df.reset_index(drop=True)

                categories = [
                    {"page": df['page'][0], "impressions": df['impressions'][0] , "clicks": df['clicks'][0]  }
                    
                ]
                n = st.slider('le nombre des pages ?', 0, len(df), len(df)//10)

                for i in range(1, n):
                    categoties = categories.append({"page": df['page'][i], "impressions": df['impressions'][i] , "clicks": df['clicks'][i]})     
                            

                
                from plotly.subplots import make_subplots


                subplots = make_subplots(
                    rows=len(categories),
                    cols=1,
                    subplot_titles=[x["page"] for x in categories],
                    shared_xaxes=True,
                    print_grid=False,
                    vertical_spacing=(0.45  / len(categories)),
                )
                _ = subplots['layout'].update(
                    width=550,
                    plot_bgcolor='#fff',
                )
                #2030a0
                for k, x in enumerate(categories):
                    for i in range(len(options)):
                        if(options[i] == 'impressions'):
                            subplots.add_trace(dict(
                                type='bar',
                                orientation='h',
                                y=[x["page"]],
                                x=[x["impressions"]],
                                text=["{:,.0f}".format(x["impressions"])],
                                hoverinfo='text',
                                textposition='auto',
                                marker=dict(
                                    color="#2030a0",
                                ),
                            ), k+1, 1)
                        if(options[i] == 'clicks'):
                            subplots.add_trace(dict(
                                type='bar',
                                orientation='h',
                                y=[x["page"]],
                                x=[x["clicks"]],
                                text=["{:,.0f}".format(x["clicks"])],
                                hoverinfo='text',
                                textposition='auto',
                                marker=dict(
                                    color="#7030a0",
                                ),
                            ), k+1, 1)
                subplots['layout'].update(
                    showlegend=False,
                )
                for x in subplots["layout"]['annotations']:
                    x['x'] = 0
                    x['xanchor'] = 'left'
                    x['align'] = 'left'
                    x['font'] = dict(
                        size=12,
                    )
                for axis in subplots['layout']:
                    if axis.startswith('yaxis') or axis.startswith('xaxis'):
                        subplots['layout'][axis]['visible'] = False

                subplots['layout']['margin'] = {
                    'l': 0,
                    'r': 20,
                    't': 20,
                    'b': 1,
                }
                height_calc = 2 * 45 * len(categories)
                height_calc = max([height_calc, 350])
                subplots['layout']['height'] = height_calc
                subplots['layout']['width'] = height_calc
                #subplots
                # Plot!
                st.plotly_chart(subplots, use_container_width=True)



            st.write("### WORLD CLOUD ")

            df = pd.read_csv('KEYWORD_PAGE_REPORT.csv')
            df = df[['date','keyword' ,'page', 'clicks' , 'impressions','position']]
            
            resultat = df.groupby(by='keyword', as_index=False).agg({'page': pd.Series.nunique})
            resultat = resultat.sort_values(by = 'page' , ascending=False , )
            resultat.to_csv('resultat.csv')
            n = st.slider('le nombre de mots ?', 0, len(resultat), len(resultat)//10)
            
            words = word(resultat , n)
            
            

            #topic = st.selectbox('select topic',['topic1','topic2','topic3'])
            #wordcloud = create_wordcloud(topic)
            
            wordcloud = WordCloud().generate(words)

            # Display the generated image:
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)

            

            st.write("### choose features per table")

            names = st.multiselect(
                "Choose tables", list(names_tables.values()), names_tables[0]
            )

            for name in names : 
                i = indexof(names_tables , name)                    
                df1 = save_data[i]
                df =pd.DataFrame(data=df1) 
                df = df.set_index(['created_timestamp'])
                df.index = pd.to_datetime(df.index)
                if not df.index.is_monotonic:
                    df = df.sort_index()

                st.write("name : ",names_tables[i])
                features = st.multiselect(
                    "Choose features", list(df.columns), list(df.columns)[len(df.columns)-1]
                )
                if not features:
                    st.error("Please select at least one feature.")
                else:
                
                    chart_data = pd.DataFrame(
                        df,
                        columns=features)

                    st.line_chart(chart_data)
        
            st.write("### choose one feature to compare with the same feature in other tables")

            
            items = itemsof(dataframe_collection)
           
            feature = st.selectbox(
                'Choose one feature to compare?',
                items)

            if feature == 'created_timestamp':
                
                st.error("Choose a valid feature.")

                
            else:
                #list(df.columns), list(df.columns)[len(df.columns)-1]
                names_ = st.multiselect(
                    "Choose tables", list(names_tables.values()), names_tables[0] , key="d"
                )

                if not names_:
                        st.error("Please select at least one feature.")
                else:

                    j = 0
                    names_1 = {}
                    dataframe_collection_1 = {}
                    for name in names_ : 
                            i = indexof(names_tables , name)                    
                        
                            items = list(dataframe_collection[i].columns)
                            if feature in items:
                                names_1[j] = name
                                dataframe_collection_1[j] = dataframe_collection[i]
                                j = j+1
                    st.write(names_1)

                    '''
                    names_1 = {}
                    for i in range(len(dataframe_collection) ):
                        names_1[i] =""
                    
                    for name in names_ : 
                            i = indexof(names_tables , name)                    
                        
                            items = list(dataframe_collection[i].columns)
                            if feature in items:
                                names_1[i] = name
                            else:
                                names_1[i] = ""
                    '''

                    
                    
                    les_min = min_(dataframe_collection_1)
                    les_max = max_(dataframe_collection_1)
                    min_max = minimax(dataframe_collection_1 , les_min , les_max)

                    for i in range(len(dataframe_collection_1)):
                        dataframe_collection_1[i] = one_feature(dataframe_collection_1[i] , feature)
                        dataframe_collection_1[i] = index_resample(dataframe_collection_1[i])
                        df3 = add_avant(min_max , les_min  , i , feature)
                        df4 = add_apres(min_max , les_max  , i , feature)
                        dataframe_collection_1[i] = add_all(dataframe_collection_1[i] , df3 , df4)
                        dataframe_collection_1[i] =drop_duplicated(dataframe_collection_1[i] , df3 , df4)


                    col={}
                    col = col_(col,feature,names_1)
                    start ={}
                    fin={}
                    startfin={}
                    startfin= start_fin(startfin, start , fin , min_max,names_1)


                    df_ = pd.DataFrame([list(startfin[0].values()), list(startfin[1].values())] , columns=list(col.values())  )
                    df_ = df_.set_index(['created_timestamp'])
                    df_.index = pd.to_datetime(df_.index )

                    df_ = df_.resample('D').mean().pad()

                    df_ = add_columns(df_,dataframe_collection_1 ,feature ,names_1)


                    chart_data = pd.DataFrame(                      
                        df_,                      
                        columns=df_.columns)

                    st.line_chart(chart_data)
            
                 
        
        except URLError as e:
            st.error(
                """
                **This app requires internet access.**
                Connection error: %s
            """
                % e.reason
            )



def show_analyse_page1(save_data , save_name):
    dataframe_collection = save_data
    names_tables = save_name
    st.title("Analyse")

    df = pd.read_csv('UMICH-SOC3.csv')
    st.write(df)

    df = df.set_index(['Date'])
    df.index = pd.to_datetime(df.index)
    if not df.index.is_monotonic:
                    df = df.sort_index()

    chart_data = pd.DataFrame(
                        df,
                        columns=['18-34' , '35-54' , '55+'])

    st.line_chart(chart_data)










    chart_data = pd.DataFrame(
                        df,
                        columns=['18-34' , '35-54' , '55+'])

    st.area_chart(chart_data)

   

    

  

    # Group data together
    hist_data = [df['18-34'], df['35-54'], df ['55+']]

    group_labels = ['18-34' , '35-54' , '55+']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)






    if(len(dataframe_collection) != 0 and len(names_tables) !=0):
       
        
        try:
            df = pd.read_csv('UMICH-SOC1.csv')

            chart_data = pd.DataFrame(
                        df,
                        columns=df.columns)

            st.area_chart(chart_data)
            '''
         names = st.multiselect(
                "Choose tables", list(names_tables.values()), names_tables[0]
            )

             for name in names : 
                i = indexof(names_tables , name)                    
                df1 = save_data[i]
                df =pd.DataFrame(data=df1) 
                df = df.set_index(['created_timestamp'])
                df.index = pd.to_datetime(df.index)
                if not df.index.is_monotonic:
                    df = df.sort_index()

                st.write("name : ",names_tables[i])
                features = st.multiselect(
                    "Choose features", list(df.columns), list(df.columns)[len(df.columns)-1]
                )
                if not features:
                    st.error("Please select at least one feature.")
                else:
                
                    chart_data = pd.DataFrame(
                        df,
                        columns=features)

                    st.area_chart(chart_data)
        '''
           
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
