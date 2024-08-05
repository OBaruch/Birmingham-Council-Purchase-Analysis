#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load fetched data
data = pd.read_pickle('Data/data.pkl')
print(data.describe())
print(data.isnull().sum())


# In[36]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autoviz.AutoViz_Class import AutoViz_Class
#get_ipython().run_line_magic('matplotlib', 'inline')
AV = AutoViz_Class()
"""df = AV.AutoViz(
    'Data/data.csv',
    sep=',', 
    depVar='',  
    dfte=None, 
    header=0,
    verbose=2,
    lowess=False,
    chart_format='png',  
    save_plot_dir='EDA/data'   
)"""


# In[4]:


# Calculate the percentage of non-null values for each column
non_null_percentage = data.notnull().mean() * 100
non_null_percentage_df = pd.DataFrame(non_null_percentage, columns=['Non-Null Percentage'])
non_null_percentage_df = non_null_percentage_df.sort_values(by='Non-Null Percentage', ascending=False)

# Adjust df
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

non_null_percentage_df


# During the initial exploratory analysis, columns with a high proportion of non-null data were identified. It was decided to remove rows containing null values in these columns to ensure data integrity. This cleaning focused on columns with more than 90% non-null data, as these represent the most complete features of the dataset.
# 
# Action:
# Rows with null values in the selected columns (more than 90% non-null) were removed. This allowed for a reduction in the dataset size without losing critical information.
# 
# Result:
# After the cleaning, the dimensions of the DataFrame changed from original dimensions to cleaned dimensions, maintaining data integrity for more accurate analysis.

# In[5]:


#Filter columns with more than 90% non-null values
columns_to_keep = non_null_percentage[non_null_percentage > 90].index

data_cleaned = data.dropna(subset=columns_to_keep)

print(f"Selected columns: {columns_to_keep.tolist()}")
print(f"Original DataFrame dimensions: {data.shape}")
print(f"DataFrame dimensions after removing rows: {data_cleaned.shape}")


# Observation:
# Columns with less than 15% non-null values were identified as highly unbalanced and not useful for the analysis. These columns have a high percentage of null values, making them unsuitable for the purpose of detecting outliers or similar analyses.
# 
# Action:
# Columns with less than 15% non-null values were removed from the dataset to ensure a more balanced and useful dataset for analysis.
# 
# Result:
# The dataset dimensions changed from original dimensions to cleaned dimensions, improving the dataset's quality and focus for further analysis.

# In[6]:


# Identify columns with less than 15% non-null values
columns_to_drop = non_null_percentage[non_null_percentage < 15].index
data_cleaned2 = data_cleaned.drop(columns=columns_to_drop)
print(f"Original DataFrame dimensions: {data_cleaned.shape}")
print(f"DataFrame dimensions after removing columns with less than 15% non-null values: {data_cleaned2.shape}")


# In[7]:


data_cleaned2.head()


# In[8]:


# Calculate the percentage of non-null values for each column
non_null_percentage2 = data_cleaned2.notnull().mean() * 100
non_null_percentage_df2 = pd.DataFrame(non_null_percentage2, columns=['Non-Null Percentage'])
non_null_percentage_df2 = non_null_percentage_df2.sort_values(by='Non-Null Percentage', ascending=False)
print(non_null_percentage_df2)


# Observation:
# The columns were reordered to follow a specific sequence for better readability and logical grouping. The new order is as follows:
# 
# TransactionDate
# Original Growth AMT
# MerchantName
# CardNumber
# TransCAC Code from 1 to 8
# The rest of the columns in descending order of their non-null percentage.
# Action:
# The columns were rearranged in the DataFrame to reflect this new order.
# 
# Result:
# The dataset is now organized with key columns at the front, followed by additional columns ordered by their percentage of non-null values, facilitating easier analysis and interpretation.

# In[9]:


# List of columns in the desired order
desired_order = [
    'TRANS DATE',
    'ORIGINAL GROSS AMT',
    'MERCHANT NAME',
    'CARD NUMBER',
    'TRANS CAC CODE 1',
    'TRANS CAC CODE 2',
    'TRANS CAC CODE 3',
    'TRANS CAC CODE 4',
    'TRANS CAC CODE 5',
    'TRANS CAC CODE 6',
    'TRANS CAC CODE 7',
    'TRANS CAC CODE 8'
]

# Get the rest of the columns ordered by non-null percentage
remaining_columns = non_null_percentage_df2.loc[~non_null_percentage_df2.index.isin(desired_order)]
remaining_columns_sorted = remaining_columns.sort_values(by='Non-Null Percentage', ascending=False).index.tolist()

# Combine the column lists in the desired order
final_order = desired_order + remaining_columns_sorted

data_reordered = data_cleaned2[final_order]
data_reordered.head()



# 

# Observation:
# The columns Transaction Tax AMT, Transoriginal Rate AMT, and Transtax Rate were identified as redundant. The Original Growth AMT column already includes the total amount with taxes, making these three columns unnecessary.
# 
# Action:
# The redundant columns (Transaction Tax AMT, Transoriginal Rate AMT, and Transtax Rate) were removed from the dataset.
# 
# Result:
# The dataset is now more streamlined, with unnecessary columns removed, simplifying further analysis.

# In[10]:


# Delate rendundant columns
columns_to_drop = ['TRANS TAX AMT', 'TRANS ORIGINAL NET AMT', 'TRANS TAX RATE']
data_cleaned_and_reordered = data_reordered.drop(columns=columns_to_drop)
data_cleaned_and_reordered.head()


# In[11]:


# Calculate the percentage of non-null values for each column
non_null_percentage3 = data_cleaned_and_reordered.notnull().mean() * 100

non_null_percentage_df3 = pd.DataFrame(non_null_percentage3, columns=['Non-Null Percentage'])
print(non_null_percentage_df3)


# In[12]:


data_cleaned_and_reordered.head()


# In[ ]:





# Observation:
# It was identified that Original Gross AMT and Billing Gross AMT are redundant, with Original Gross AMT containing 100% of the data.
# 
# Action:
# The Billing Gross AMT column was removed from the dataset to avoid redundancy.
# 
# Result:
# The dataset is now more streamlined, retaining only the Original Gross AMT column which contains complete data.
# 
# 

# In[13]:


new_data = data_cleaned_and_reordered.drop(columns=['BILLING GROSS AMT'])
new_data.head()


# Observation:
# The dataset columns were analyzed to determine their data types and the distribution of data types within each column. This information was used to decide on the appropriate method for filling null values.
# 
# Action:
# 
# Numerical columns: Filled null values with the mean of the column.
# Categorical columns: Filled null values with the string "empty".
# Result:
# The dataset now has all null values appropriately filled, ensuring consistency and completeness for further analysis.

# Observation:
# The columns in the dataset were adjusted to the appropriate data types, and null values in categorical columns were replaced with the text "NULL".
# 
# Action:
# 
# TRANSACTION DATE was converted to date type.
# ORIGINAL GROSS AMT was converted to float type.
# MERCHANT NAME, CARD NUMBER, TRANS CAC CODE 1-8, DIRECTORATE, TRANS CAC DESC 1-2, ORIGINAL CUR, BILLING CUR CODE, TRANS VAT DESC, and TRANS TAX DESC were converted to categorical type.
# Null values in categorical columns were replaced with "NULL".
# Result:
# The dataset now has appropriately typed columns with categorical columns having no null values.

# In[14]:


import pandas as pd

# last data values
data = new_data

# Cambiar el tipo de datos y manejar valores nulos
data['TRANS DATE'] = pd.to_datetime(data['TRANS DATE'], errors='coerce')
data['ORIGINAL GROSS AMT'] = data['ORIGINAL GROSS AMT'].astype(float)
data['MERCHANT NAME'] = data['MERCHANT NAME'].astype('category')
data['CARD NUMBER'] = data['CARD NUMBER'].apply(lambda x: str(x)[-4:] if pd.notnull(x) else x).astype('category')
for i in range(1, 9):
    data[f'TRANS CAC CODE {i}'] = data[f'TRANS CAC CODE {i}'].astype('category')
data['Directorate'] = data['Directorate'].astype('category')
data['TRANS CAC DESC 1'] = data['TRANS CAC DESC 1'].astype('category')
data['TRANS CAC DESC 2'] = data['TRANS CAC DESC 2'].astype('category')
data['ORIGINAL CUR'] = data['ORIGINAL CUR'].astype('category')
data['BILLING CUR CODE'] = data['BILLING CUR CODE'].astype('category')
data['TRANS VAT DESC'] = data['TRANS VAT DESC'].astype('category')
data['TRANS TAX DESC'] = data['TRANS TAX DESC'].astype('category')

# Add the 'NULL' category to categorical columns and replace null values
categorical_columns = data.select_dtypes(['category']).columns
for column in categorical_columns:
    data[column] = data[column].cat.add_categories('NULL')
    data[column] = data[column].fillna('NULL')


print(data.dtypes)


# In[15]:


data.head()


# # Guardar datos limpios

# In[16]:


# Save the cleaned data to a CSV file
data.to_csv('Data/data_clean.csv', index=False)
# Save the cleaned data to a pickle file
data.to_pickle('Data/data_celan.pkl')


# # EDA WHIT CLEAN DATA

# In[ ]:


from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
"""df = AV.AutoViz(
    'Data/data_clean.csv',
    sep=',', 
    depVar='',  
    dfte=None, 
    header=0,
    verbose=2,
    lowess=False,
    chart_format='png',  
    save_plot_dir='EDA/data_clean'   
)"""


# In[38]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('Data/data_clean.csv')

# Fill missing values with 'NULL'
df.fillna('NULL', inplace=True)

# Impute missing values in specific columns with a constant
columns_to_impute = [
    'MERCHANT NAME', 'Directorate', 'TRANS CAC DESC 1', 
    'TRANS CAC DESC 2', 'BILLING CUR CODE', 'ORIGINAL CUR', 
    'TRANS VAT DESC', 'TRANS TAX DESC'
]
trans_cac_columns = [f'TRANS CAC CODE {i}' for i in range(1, 9)]
columns_to_impute.extend(trans_cac_columns)

for col in columns_to_impute:
    df[col].replace('NULL', '123', inplace=True)

# Convert 'ORIGINAL GROSS AMT' to numeric and cap outliers
df['ORIGINAL GROSS AMT'] = pd.to_numeric(df['ORIGINAL GROSS AMT'], errors='coerce')
q_low = df['ORIGINAL GROSS AMT'].quantile(0.01)
q_high = df['ORIGINAL GROSS AMT'].quantile(0.99)
df['ORIGINAL GROSS AMT'] = df['ORIGINAL GROSS AMT'].apply(lambda x: q_low if x < q_low else q_high if x > q_high else x)

# Encode categorical columns
categorical_columns = ['MERCHANT NAME', 'Directorate', 'TRANS VAT DESC', 'TRANS TAX DESC'] + trans_cac_columns

label_encoders = {}
for col in categorical_columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = label_encoder

# Save the encoders to a pickle file
with open('Encoders/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Handle rare categories in 'TRANS TAX DESC'
rare_categories = ['0 Percent', 'VE', 'VF', 'VT', 'VS', '6.1 Percent', '12.5%']
df['TRANS TAX DESC'] = df['TRANS TAX DESC'].apply(lambda x: 'Other' if x in rare_categories else x)


# Drop the 'TRANS TAX DESC' column - High correlation to  ['TRANS VAT DESC']
df.drop(columns=['TRANS TAX DESC'], inplace=True)

# Group rare categories in 'ORIGINAL CUR' under 'Foreign Coin'
# Categorize 'ORIGINAL CUR' into three categories: 'GBP', '123', and 'OTHERS'
df['ORIGINAL CUR'] = df['ORIGINAL CUR'].apply(lambda x: 'GBP' if x == 'GBP' else ('123' if x == '123' else 'OTHERS'))


# Save the treated dataset
df.to_csv('Data/data_clean_treated.csv', index=False)



# In[34]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autoviz.AutoViz_Class import AutoViz_Class
#get_ipython().run_line_magic('matplotlib', 'inline')
AV = AutoViz_Class()
"""df = AV.AutoViz(
    'Data/data_clean_treated.csv',
    sep=',', 
    depVar='',  
    dfte=None, 
    header=0,
    verbose=2,
    lowess=False,
    chart_format='png',  
    save_plot_dir='EDA/data_clean_treated'   
)"""



# # Normalize data

# In[20]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Data/data_clean_treated.csv')


# Normalize the ORIGINAL GROSS AMT column
df['ORIGINAL GROSS AMT'] = (df['ORIGINAL GROSS AMT'] - df['ORIGINAL GROSS AMT'].mean()) / df['ORIGINAL GROSS AMT'].std()

# Save the treated dataset
df.to_csv('Data/data_clean_treated_normalized.csv', index=False)



# In[33]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autoviz.AutoViz_Class import AutoViz_Class
#get_ipython().run_line_magic('matplotlib', 'inline')
AV = AutoViz_Class()
df = AV.AutoViz(
    'Data/data_clean_treated_normalized.csv',
    sep=',', 
    depVar='',  
    dfte=None, 
    header=0,
    verbose=2,
    lowess=False,
    chart_format='png',  
    save_plot_dir='EDA/data_clean_treated_normalized'   
)




# In[22]:


df.describe()


# In[24]:


df.dtypes


# In[25]:


df.head()

