# Author: Spencer Chew
# version: 1.0
# July 2023
# pip install streamlit --user
# streamlit version
# streamlit run <Filepath>


import streamlit as st
from PIL import Image
# other libs

# -- Set page config
apptitle = 'Capstone Project'

st.set_page_config(page_title=apptitle, page_icon='⌚🌡️', layout= 'wide', initial_sidebar_state="expanded")
# random icons in the browser tab

#Where your image is 
image = Image.open('Wearable2.jpg')
st.image(image, width=450)


st.subheader('**Project Summary**')
# A brief description
st.write('This project is focused on the prediction risk of ♨️ heat exhaustion from multivariate sensor data collected from consumer-grade wearables like ⌚ smartwatches')


st.subheader('**Objectives**')
# A brief description

st.markdown("- To predict risk of heat exhaustion from multivariate sensor data collected from consumer-grade wearables like smartwatches")
st.markdown("- To develop a novel model for heat stroke prediction as no-known research to predict heat tolerance is currently available from wearable devices")


st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)
