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
- Draws on data from 66 pariticpants from a San Diego State Study (https://data.mendeley.com/datasets/g5sb382smp/1)
''')


#Caching the model for faster loading
#@st.cache

#####################
#Data Prep

# Load the dataset into a Pandas dataframe
df = pd.read_csv("Survival_Analysis v2a.csv")

#rename critical columns
df = df.rename(columns={"('Time', 'max')": "time_to_event",
                        'Resting HR': 'resting_hr',
                        'BP (Systolic)': 'bp_systolic',
                        'BP (Diastolic)': 'bp_diastolic',
                        'VO2 (relative)': 'vo2_relative',
                        'VO2 (absolute)': 'vo2_absolute' ,
                        'BF %':'body_fat_perc',
                        # 'Weight (Kg)':'Weight (kg)',
                        'Av. Temp': 'avr_temperature',
                        'Av. Humidity':'avr_humidity',
                                    'predicted BT value':'predicted_BT',
                                    'predicted HR value':'predicted_HR'})

# get the number of rows and columns
num_rows, num_cols = df.shape

#print("Number of rows:", num_rows)
#print("Number of columns:", num_cols)

#st.write('Dataset contains', num_rows, 'rows'
 #   ,'Dataset contains', num_cols, 'columns' )

#st.dataframe(df,use_container_width = True)

# define BMI bins or categories based on WHO classification
bmi_bins = pd.cut(df['BMI'], bins=[0, 25, df['BMI'].max()], labels=['Not overweight', 'Overweight'])

# add BMI bins as a new column to the dataframe
# df['bmi_bins_full'] = bmi_bins_full
df['bmi_bins'] = bmi_bins
df['bmi_bins'].value_counts()

#Create interaction variable for BP
df['bp_systolic_diastolic'] = df['bp_systolic'] * df['bp_diastolic']
df['bp_systolic_diastolic'].head()

# one-hot encode categorical variables
cat_vars = ['Gender','bmi_bins']
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[cat_vars]).toarray()
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_vars))

#avr_temperature and avr_humidity columns has 2 null values each. impute with avr value of column
# Calculate the average value of avr_temperature column
average_value_temp = df['avr_temperature'].mean()
# Impute the average value for NaNs in the column
df['avr_temperature'].fillna(average_value_temp, inplace=True)

# Calculate the average value of avr_humidity column
average_value_humidity = df['avr_humidity'].mean()
# Impute the average value for NaNs in the column
df['avr_humidity'].fillna(average_value_humidity, inplace=True)

#####################
######################## section-1 ##################
# Let's add a sub-title

# Let's load and display a data set
st.subheader('*1. Enter information below*')

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
