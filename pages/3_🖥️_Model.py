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
df = pd.read_csv("Survival_Analysis v2a.csv")

#rename critical columns
df = df.rename(columns={"('Time', 'max')": "time_to_event",
                        'Resting HR': 'resting_hr',
                        'BP (Systolic)': 'bp_systolic',
                        'BP (Diastolic)': 'bp_diastolic',
                        'VO2 (relative)': 'vo2_relative',
                        'VO2 (absolute)': 'vo2_absolute' ,
                        'BF %':'body_fat_perc',
                        'Av. Temp': 'avr_temperature',
                        'Av. Humidity':'avr_humidity',
                        'predicted BT value':'predicted_BT',
                        'predicted HR value':'predicted_HR'})

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

# combine one-hot encoded variables with df
df = pd.concat([encoded_df, df], axis=1)

#avr_temperature and avr_humidity columns has 2 null values each. impute with avr value of column
# Calculate the average value of avr_temperature column
average_value_temp = df['avr_temperature'].mean()
# Impute the average value for NaNs in the column
df['avr_temperature'].fillna(average_value_temp, inplace=True)

average_value_humidity = df['avr_humidity'].mean()
# Impute the average value for NaNs in the column
df['avr_humidity'].fillna(average_value_humidity, inplace=True)

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

df_cph = df.copy()
df_cph = df_cph[variables_corrplot]
df_cph.columns

# calculate correlation matrix
corr_matrix = df_cph.corr()

# create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)



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

variables_vif = [
    'Gender_F',
    'Gender_M',
    'bmi_bins_Not overweight',
    'bmi_bins_Overweight',
    'Age',
    'Height (m)',
    'Weight (Kg)',
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

temp=df[variables_vif]
temp.columns


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

compute_vif(temp,variables_vif).sort_values('VIF', ascending=False)

moreThan5=True
while moreThan5:
  val=compute_vif(temp,variables_vif).sort_values('VIF', ascending=False).head(1)['VIF'].values[0]
  if val>5:
    column=compute_vif(temp,variables_vif).sort_values('VIF', ascending=False).head(1)['Variable'].values[0]
    print("drop column: {}, VIF: {}".format(column,val))
    variables_vif.remove(column)
  else:
    moreThan5=False

compute_vif(temp,variables_vif).sort_values('VIF', ascending=False)

#Manually deciding to drop Height as Height is in BMI
columns_to_drop = ['Height (m)']
variables_vif = [col for col in variables_vif if col not in columns_to_drop]

temp = temp[variables_vif]

compute_vif(temp,variables_vif).sort_values('VIF', ascending=False)

###COX###

#use list of finalised variables variables_vif from above after removing VIF >5 for cox model
variables_cox_nonpca = variables_vif.copy()
variables_cox_nonpca.append('Heat Stroke')
variables_cox_nonpca.append('time_to_event')
print(variables_cox_nonpca)

df_cph_vif = df.copy()
df_cph_vif = df_cph_vif[variables_cox_nonpca]

from sklearn.utils import resample

## Re-structure dataframe to have 85% non-heatstroke, 15% heatstroke
# Identify majority and minority classes
majority_class = df_cph_vif[df_cph_vif['Heat Stroke'] == 0]
minority_class = df_cph_vif[df_cph_vif['Heat Stroke'] == 1]

# Randomly select 240 rows from the majority class
n_samples = int((minority_class.shape[0]/0.15)-minority_class.shape[0])
selected_majority = resample(majority_class, n_samples=n_samples, replace=False, random_state=22)

print("Heatstroke = 0 samples: " + str(selected_majority.shape[0]))
print("Heatstroke = 1 samples: " + str(minority_class.shape[0]))

df_cph_vif = selected_majority.append(minority_class, ignore_index=True)
print("Total number of samples: " + str(df_cph_vif.shape[0]))

df_cph_vif

# split data into train and test sets
train_data, test_data = train_test_split(df_cph_vif, test_size=0.2, random_state=42)

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
df_cph_riskscore = df_cph_vif[cox_nonpca_features]

# Calculate predicted hazard ratios for each participant
#computes linear combination of the predictor variables with their corresponding coefficients and take exponential
predicted_hazard_ratios = np.exp(np.dot(df_cph_riskscore.values, cph.params_))

# Transform hazard ratios into risk scores (logarithmic transformation)
risk_scores = np.log(predicted_hazard_ratios)

# Scale risk scores to a specific range (e.g., 0-100)
min_score = min(risk_scores)
max_score = max(risk_scores)
scaled_risk_scores = 100 * (risk_scores - min_score) / (max_score - min_score)

# Add risk scores to the original dataset (variables_cox_nonpca)
df_cph_riskscore["Risk_Score"] = scaled_risk_scores
df_cph_riskscore
#Get Range of Values for each variable
# Assuming you have a DataFrame called 'df_cph_riskscore '
result_dict = {}  # Dictionary to store column names as keys and max/min values as values
df_rs = df_cph_riskscore

for column in df_rs.columns:
    max_value = df_rs[column].max()
    min_value = df_rs[column].min()
    pct_25 = df_rs[column].quantile(0.25)
    pct_50 = df_rs[column].quantile(0.50)
    pct_75 = df_rs[column].quantile(0.75)

    result_dict[column] = {'Max': max_value, 'Min': min_value,'P25': pct_25, 'P50': pct_50, 'P75': pct_75 }

result_df = pd.DataFrame.from_dict(result_dict, orient='index')


sorted_summary.reset_index(drop=False, inplace=True)

result_df.reset_index(drop=False, inplace=True)

# Get all column names
column_names = list(result_df.columns.values)
column_names_2 = list(sorted_summary .columns.values)

result_df['index'] = result_df['index'].astype(str)
sorted_summary['coef'] = sorted_summary['coef'].astype(str)

result_df

predicted_hazard_ratios_participant = np.exp(np.dot(result_df.values, cph.params_))

# Transform hazard ratios into risk scores (logarithmic transformation)
risk_scores_participant = np.log(predicted_hazard_ratios_participant)

# Scale risk scores to a specific range (e.g., 0-100)
min_score = min(risk_scores)
max_score = max(risk_scores)
scaled_risk_scores = 100 * (risk_scores_participant - min_score) / (max_score - min_score)

scaled_risk_scores

#sorted_summary['coef']

# Print the column names
column_names
column_names_2

#merged_df = sorted_summary.merge(result_df, left_on='covariate', right_on='index')

# Display the result DataFrame
#merged_df

######################## section-1 ##################
# Let's add a sub-title

# Let's load and display a data set
st.subheader('*1. Enter information below*')

#df1 = pd.DataFrame()


# Create a form with input fields
with st.form("Input Form"):
    inputs = {}

    for index, row in merged_df.iterrows():
        variable = row['covariate']
        min_value = row['Min']
        max_value = row['Max']
        P25 = row['P25']

        inputs[variable] = st.number_input(f"{variable}:",
                                           value=P25, key=variable)

    # Create a submit button
    submitted = st.form_submit_button("Submit")

########################## section-2 #####################
st.subheader('*2. Section 2**')
# Process the form data when the button is clicked
if submitted:
    # Calculate risk score using coefficients and input values
    risk_score = 0.0

    for index, row in merged_df.iterrows():
        variable = row['covariate']
        coef = row['coef']
        input_value = inputs[variable]

        risk_score += float(coef) * float(input_value)

    # Display the risk score
    st.write("Risk Score:", risk_score)

    # Visualize the risk score as a bar chart
    fig, ax = plt.subplots()
    ax.bar("Risk Score", risk_score)
    ax.set_xlabel("Category")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk Score Visualization")

    # Display the chart in Streamlit
    st.pyplot(fig)
#st.dataframe(df1.style.highlight_max(axis=0))
#st.write('source: https://docs.streamlit.io/en/stable/api.html#display-data')



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
