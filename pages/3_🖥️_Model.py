import streamlit as st
from PIL import Image
# other libs

# -- Set page config
apptitle = 'Capstone Project'

st.set_page_config(page_title=apptitle, page_icon='random', layout= 'wide', initial_sidebar_state="expanded")
# random icons in the browser tab

#File Path for your image
image = Image.open('Wearable2.jpg')
#st.image(image, width=450)

#####################
# Header 
st.title('EBAC Capstone Project')


st.markdown('## Risk Scoring Tool', unsafe_allow_html=True)
st.info('''
- This risk scoring tool predicts the likelihood of heat injury based on various indicators shown below
- Draws on data from 66 pariticpants from a San Diego State Study
''')


#Caching the model for faster loading
#@st.cache


######################## section-1 ##################
# Let's add a sub-title
st.write('Project Details')


# Let's load and display a data set
st.subheader('*1. Section 1*')

#df1 = pd.DataFrame()


resting_hr = st.number_input('resting_hr:', min_value=0.1, max_value=100.0, value=1.0)

bp_systolic_diastolic = st.number_input('bp_systolic_diastolic:', min_value=0.1, max_value=10.0, value=1.0)

Age = st.number_input('Age:', min_value=0.1, max_value=100.0, value=1.0)

Gender = st.selectbox('Gender', ['Male', 'Female'])

vo2_relative = st.number_input('vo2_relative:', min_value=0.1, max_value=10.0, value=1.0)

Temperature = st.number_input('Temperature:', min_value=0.1, max_value=50.0, value=1.0)

Heart_Rate  = st.number_input('Heart Rate:', min_value=0.1, max_value=200.0, value=1.0)


#st.dataframe(df1.style.highlight_max(axis=0))
#st.write('source: https://docs.streamlit.io/en/stable/api.html#display-data')

########################## section-2 #####################
st.subheader('*2. Section 2**')
#boston = datasets.load_boston()
#df2 = pd.DataFrame(boston.data, columns=boston.feature_names)
# st.dataframe(df2)

# let us try some plotting
#fig, ax = plt.subplots(figsize=(6, 3))
# sns.boxplot(data=df2)
# st.pyplot(fig)



########################## section-3 ######################################
# try to load diabetes dataset and plot histogram for age of patients

st.subheader('*3. Section 3**')

# st.dataframe(df3)


# st.table(df3)


# st.write(df3)

# df4 = pd.DataFrame({
# 	'first column': [1, 2, 3, 4],
# 	'second column': [5, 6, 7, 8]
# 	})
# if st.checkbox('show/hide data'):
# 	df4

# option = st.selectbox(
# 	'which number do you like best?', df4['second column'])
# st.write(f'You selected: {option}')

# option = st.selectbox(
# 	'which number do you like best?', 
# 	df4.unstack().reset_index(drop=True))
# st.write(f'Now, you selected: {option}')

# chosen = st.radio(
# 	'ISS new courses',
# 	('DSSI', 'XAI', 'AMLFin', 'Credit Scoring'))

# st.write(f'You opted to learn: {chosen}')


# map_data = pd.DataFrame(
# 	np.random.randn(100, 2)/ [20, 20] + [1.3521, 103.8198], 
# 	columns = ['lat', 'lon'])
# st.map(map_data)


########################## sidebar ######################################
# try to load diabetes dataset and plot histogram for age of patients

#with st.sidebar:
   # with st.echo():
        #st.write("This code will be printed to the sidebar.")
