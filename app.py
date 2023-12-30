import streamlit as st
import pandas as pd
import os



# import pandas profilling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# import ML stufff
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar :
    st.image("https://st4.depositphotos.com/1029305/30841/i/450/depositphotos_308415734-stock-photo-persian-green-glossy-geometrical-letter.jpg")
    st.title("my first app")
    choice= st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipline using Streamlit, Pandas profiling and PyCaret and it's awesome!")
    
if os.path.exists("sourcecode.csv") :
    df= pd.read_csv("sourcecode.csv", index_col= None)

if choice == "Upload":
    st.title("data Modelling!")
    st.text("Upload your data modelling her ")
    st.markdown("---")
    file= st.file_uploader("Upload Your Dataset Here")
    
    if file :
        df= pd.read_csv(file, index_col= None)
        df.to_csv("sourcecode.csv", index= None)
        st.dataframe(df)

if choice == "Profiling" :
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

if choice == "ML" :
    st.title("Machine Learning %%%%")
    target= st.selectbox("Select Your Target", df.columns)
    if st.button("Train model") :
        setup(df, target= target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download" :
    with open("best_model.pkl", "rb") as f :
        st.download_button("Download the Model", f, "trained_model.pkl")
        
