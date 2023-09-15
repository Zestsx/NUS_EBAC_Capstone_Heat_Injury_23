import streamlit as st
import pandas as pd
import altair as alt

#Import relevant packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Visualisation", page_icon="📊", layout= 'wide', initial_sidebar_state="expanded")

st.markdown("# Data Visualisation")
st.sidebar.header("Data Visualisation")
st.write(
    """Data Visualisation"""
)


#Insert code to show dataframe

# Load the dataset into a Pandas dataframe
df = pd.read_csv("Survival_Analysis v2a.csv")

#rename critical columns
df = df.rename(columns={"('Time', 'max')": "time_to_event",
                        "('Time', 'min')": "min_time",
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

df

df2 = df

df = df[df['min_time'] == 0]

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

#print(df_cph)

# calculate correlation matrix
#corr_matrix = df_cph.corr()

# create a heatmap of the correlation matrix
#fig, ax = plt.subplots(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

# show the plot
#plt.show()

st.subheader('**Exploratory Data Visualisation**')
# A brief description

# Binning Age into categories
df['Age_Category'] = pd.cut(df['Age'], bins=[0,10,20, 30, 40, 50, 60,70], labels=['0-10','10-19','20-29', '30-39', '40-49', '50-59','60-69'])

# Plotting Age Category
#age_counts = df['Age_Category'].value_counts().sort_index()
#plt.bar(age_counts.index, age_counts.values)
#plt.xlabel('Age Category')
#plt.ylabel('Count')
#plt.title('Distribution of Age Categories')
#plt.show()

#st.markdown("- Placeholder")
#st.markdown("- Placeholder")

# Binning Height into categories
df['Height_Category'] = pd.cut(df['Height (m)'], bins=[1.50, 1.60, 1.70, 1.80, 1.90], labels=['150-159', '160-169', '170-179', '180-189'])

# Binning Weight into categories
df['Weight_Category'] = pd.cut(df['Weight (Kg)'], bins=[50, 60, 70, 80, 90], labels=['50-59', '60-69', '70-79', '80-89'])

# Visualize Age Category
age_counts = df['Age_Category'].value_counts().reset_index()
age_counts.columns = ['Age_Category', 'Count']
fig_age = px.bar(age_counts, x='Age_Category', y='Count')
st.plotly_chart(fig_age)
st.write('*Age Breakdown*')
st.write("---") 

# Visualize Height Category
height_counts = df['Height_Category'].value_counts().reset_index()
height_counts.columns = ['Height_Category', 'Count']
fig_height = px.bar(height_counts, x='Height_Category', y='Count')
st.plotly_chart(fig_height)
st.write('*Height Breakdown*')
st.write("---") 

# Visualize Weight Category
weight_counts = df['Weight_Category'].value_counts().reset_index()
weight_counts.columns = ['Weight_Category', 'Count']
fig_weight = px.bar(weight_counts, x='Weight_Category', y='Count')
st.plotly_chart(fig_weight)
st.write('*Weight Breakdown*')
st.write("---") 

# Create a gender pie chart using plotly
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
fig_gender = px.pie(gender_counts, values='Count', names='Gender')
st.plotly_chart(fig_gender)
st.write('*Gender*')
st.write("---")

# Create a scatter histogram using Plotly Express
fig_bt_hr = px.scatter(df, x='predicted_BT', y='predicted_HR')
st.plotly_chart(fig_bt_hr)
st.write('*Heart Rate & Body Temperature*')
st.write("---")

# Create a partition using Plotly Express
df2['max_value_within_partition'] = df2.groupby('Subject_ID')['time_to_event'].transform('max')

# Filter the dataframe for rows with the highest value within each partition
filtered_df = df2[df2['time_to_event'] == df2['max_value_within_partition']]

filtered_df

# Create a scatter histogram using Plotly Express
fig_bt_hr = px.scatter(filtered_df, x='predicted_BT', y='predicted_HR',color='Heat Stroke')
fig_bt_hr.update_layout(xaxis_range=[min(filtered_df['predicted_BT']) - 1, max(filtered_df['predicted_BT']) + 1],
                        yaxis_range=[min(filtered_df['predicted_HR']) - 5, 200)
st.plotly_chart(fig_bt_hr)
st.write('*Heart Rate & Body Temperature*')
st.write("---")

# Customize the axis ranges
# Exclude outliers



# create a heatmap of the correlation matrix
#fig, ax = plt.subplots(figsize=(8, 6))
#sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

# show the plot
