import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lifelines import *
from lifelines.utils import k_fold_cross_validation
import matplotlib.pyplot as plt
import seaborn as sns

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
#Data Prep, Modelling

# Load the dataset into a Pandas dataframe
df_nopca = pd.read_csv("Survival_Analysis v2a.csv")

#rename critical columns
df_nopca = df_nopca.rename(columns={"('Time', 'max')": "time_to_event",
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

# define BMI bins or categories based on WHO classification
# bmi_bins_full = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, df['BMI'].max()], labels=['Underweight', 'Normal weight', 'Overweight', 'Obese'])
bmi_bins = pd.cut(df_nopca['BMI'], bins=[0, 25, df_nopca['BMI'].max()], labels=['Not overweight', 'Overweight'])

# add BMI bins as a new column to the dataframe
# df['bmi_bins_full'] = bmi_bins_full
df_nopca['bmi_bins'] = bmi_bins
df_nopca['bmi_bins'].value_counts()

#Create interaction variable for BP
df_nopca['bp_systolic_diastolic'] = df_nopca['bp_systolic'] * df_nopca['bp_diastolic']
df_nopca['bp_systolic_diastolic'].head()

# one-hot encode categorical variables
cat_vars = ['Gender','bmi_bins']
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df_nopca[cat_vars]).toarray()
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_vars))

# combine one-hot encoded variables with df
df_nopca = pd.concat([encoded_df, df_nopca], axis=1)

#avr_temperature and avr_humidity columns has 2 null values each. impute with avr value of column
# Calculate the average value of avr_temperature column
average_value_temp = df_nopca['avr_temperature'].mean()
# Impute the average value for NaNs in the column
df_nopca['avr_temperature'].fillna(average_value_temp, inplace=True)

# Calculate the average value of avr_humidity column
average_value_humidity = df_nopca['avr_humidity'].mean()
# Impute the average value for NaNs in the column
df_nopca['avr_humidity'].fillna(average_value_humidity, inplace=True)

#plotting corr matrix
variables_corrplot = [
    'Age',
    'Gender',
    'Height (m)',
    'Weight (Kg)',
    'BMI',
    'vo2_relative',
    'vo2_absolute',
    'resting_hr',
    'bp_systolic_diastolic',
    'body_fat_perc',
    'avr_temperature',
    'avr_humidity',
    'predicted_BT',
    'predicted_HR'
]

df_cph = df_nopca.copy()
df_cph = df_cph[variables_corrplot]
df_cph.columns

# calculate correlation matrix
corr_matrix = df_cph.corr()

# create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

# show the plot
plt.show()


variables_nonpca = [
    'Age',
    'Gender',
    'Height (m)',
    'Weight (Kg)',
    'BMI',
    'vo2_relative',
    'vo2_absolute',
    'resting_hr',
    'bp_systolic_diastolic',
    'body_fat_perc',
    'avr_temperature',
    'avr_humidity',
    'predicted_BT',
    'predicted_HR'
]

temp=df_nopca[variables_nonpca]
temp.head()

# convert category(str) to numerical data
temp['Gender'] = temp['Gender'].map({'M':0, 'F':1})

# compute the vif for all given features
def compute_vif(temp_df,considered_features):

    X = temp_df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


compute_vif(temp,variables_nonpca).sort_values('VIF', ascending=False)

# drop value because it is >5
variables_nonpca.remove('Weight (Kg)')

compute_vif(temp,variables_nonpca).sort_values('VIF', ascending=False)

# drop value because it is >5
variables_nonpca.remove('vo2_absolute')

compute_vif(temp,variables_nonpca).sort_values('VIF', ascending=False)

# drop value because it is >5
variables_nonpca.remove('Gender')

compute_vif(temp,variables_nonpca).sort_values('VIF', ascending=False)

variables_cox_nonpca = [
    'time_to_event',
    'Heat Stroke',
    'Age',
    'Gender_F',
    'Gender_M',
    'bmi_bins_Not overweight',
    'bmi_bins_Overweight',
    'Height (m)',
    'Weight (Kg)',
#     'BMI',
    'vo2_relative',
    'vo2_absolute',
    'resting_hr',
    'bp_systolic_diastolic',
    'body_fat_perc',
    'avr_temperature',
    'avr_humidity',
    'predicted_BT',
    'predicted_HR'
]

df_cph = df_nopca.copy()
df_cph = df_cph[variables_cox_nonpca]


compute_vif(df_cph,variables_cox_nonpca).sort_values('VIF', ascending=False)

moreThan5=True
while moreThan5:
  val=compute_vif(df_cph,variables_cox_nonpca).sort_values('VIF', ascending=False).head(1)['VIF'].values[0]
  if val>5:
    column=compute_vif(df_cph,variables_cox_nonpca).sort_values('VIF', ascending=False).head(1)['Variable'].values[0]
    print("drop column: {}, VIF: {}".format(column,val))
    variables_cox_nonpca.remove(column)
  else:
    moreThan5=False

df_cph=df_cph[variables_cox_nonpca]

# split data into train and test sets
train_data, test_data = train_test_split(df_cph, test_size=0.2, random_state=42)

#TRAIN
# create the proportional hazards model
cph = CoxPHFitter(penalizer = 0.1)

target = 'time_to_event'
event = 'Heat Stroke'

# fit the model on the training data, specifying model effects
cph.fit(train_data, duration_col=target, event_col=event)

# print the TRAIN coefficients with p-values

# evaluate the model
concordance_index = cph.score(train_data, scoring_method='concordance_index')
print('Concordance index for TRAIN:', concordance_index)

summary = cph.summary
summary['Significant'] = ['Y' if p < 0.05 else 'N' for p in summary['p']]

sorted_summary = summary.sort_values(by='exp(coef)', ascending=False)
sorted_summary

cph.check_assumptions(train_data, p_value_threshold=0.05, show_plots=True)
#Proportional hazard assumption looks okay.

# TEST

# fit the model on the test data, specifying model effects
cph.fit(test_data, duration_col=target, event_col=event)

# print TEST coefficients with p-values

# evaluate the model
concordance_index = cph.score(test_data, scoring_method='concordance_index')
print('Concordance index for TEST:', concordance_index)

summary = cph.summary
summary['Significant'] = ['Y' if p < 0.05 else 'N' for p in summary['p']]

sorted_summary = summary.sort_values(by='exp(coef)', ascending=False)
sorted_summary

#####################
#Risk Scoring

# List of columns to drop
columns_to_drop = ['time_to_event', 'Heat Stroke']

# Create a new list with columns that are not in columns_to_drop
cox_nonpca_features = [col for col in variables_cox_nonpca if col not in columns_to_drop]

#Get Range of Values for each variable

# Assuming you have a DataFrame called 'df'

result_dict = {}  # Dictionary to store column names as keys and max/min values as values

df_rs = df_cph[cox_nonpca_features]

for column in df_rs.columns:
    max_value = df_rs[column].max()
    min_value = df_rs[column].min()

    result_dict[column] = {'Max': max_value, 'Min': min_value}

result_df = pd.DataFrame.from_dict(result_dict, orient='index')

result_df.reset_index(drop=False, inplace=True)

# Get all column names
column_names = list(result_df.columns.values)
column_names_2 = list(sorted_summary .columns.values)

# Print the column names
column_names
column_names_2

# Perform the left join and merge based on a common key column
merged_df = sorted_summary.merge(result_df, left_on='coef', right_on='index', how='left')

# Display the result DataFrame
merged_df

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
