import streamlit as st
from src.extractor import summarizer
from src.utils import cleanhtml



st.set_page_config(
    layout="wide",
    page_title="Abstractive Summarizer AI",
    page_icon="🗞️",)

st.title('🗞️Abstractive Summarizer AI')

description = st.text_area('Description', placeholder= 'Enter Content of News')
col1, col2 = st.columns(2)
with col1:
    min_length = st.slider('Summary min length', min_value = 100, max_value = 200, step = 25)
with col2:
    max_length = st.slider('max length', min_value = 200, max_value = 500, step = 25)

if(st.button('Generate Summary')):
    st.text_area('BART Summary: ', value = summarizer(cleanhtml(description),'BART',min_length, max_length))
    st.text_area('T5-base Summary: ', value = summarizer(cleanhtml(description),'T5-base',min_length, max_length))

