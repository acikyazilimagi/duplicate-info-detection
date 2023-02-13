import streamlit as st
import pandas as pd
import requests


@st.cache_data
def covert_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

try:
    uploaded_file = st.file_uploader("CSV Dosyasi yukle.", type="csv")
except Exception as e:
    st.warning("Dosya yüklenemedi. Lütfen tekrar deneyin.")
    st.error(e)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # response = requests.post("http://localhost:5000/process", files={"file": uploaded_file})
    # response = response.json()
    # if response.status_code == 200:
    #     st.write("Response from the API call")
    #     st.write(response)

    csv = covert_to_csv(df)

    st.download_button(
        'CSV dosyasi indir',
        csv,
        'preprocessed_addresses.csv',
        'text/csv',
        key='download-csv'
    )
