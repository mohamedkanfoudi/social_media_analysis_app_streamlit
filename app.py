from select import select
import streamlit as st
import pandas as pd
from predict_future_values_page import show_predict_future_values_page
from sentiment_analysis_page import show_sentiment_analysis_page
from analyse_page import show_analyse_page
from statement_predict import show_statement_predict
from show_page import show_show_page , save_data , save_name
page = st.sidebar.selectbox("Prepare data to analyse and to predict future values", ("Connect to Snowflake" ,"Analyse","Predict future values","Sentiment analysis for comments"))

dataframe_collection = {} 
names_tables = {}
dataframe_collection = save_data()
names_tables = save_name()

if page == "statement predict":
    show_statement_predict()
elif page == "Connect to Snowflake" : 
    show_show_page(dataframe_collection , names_tables)
elif page == "Analyse" : 
    show_analyse_page(dataframe_collection , names_tables)
elif page == "Predict future values" : 
    show_predict_future_values_page(dataframe_collection , names_tables)
elif page == "Sentiment analysis for comments" : 
    show_sentiment_analysis_page(dataframe_collection , names_tables)

    