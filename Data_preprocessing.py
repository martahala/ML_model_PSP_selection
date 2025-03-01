''' Data understanding and preparation '''
#Uploading the given dataset using Pandas 
import pandas as pd
df = pd.read_excel('') #Insert your file path

#Renaming the column 'Unnamed: 0'
df.rename(columns={'Unnamed: 0': 'Transaction_ID'}, inplace=True)

#Checking the columns for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n",missing_values)

#Getting the percentage of missing values for each column
missing_percentage = (df.isnull().sum() / len(df))*100
print("The percentage of missing values in each column is:\n", missing_percentage)

#Checking the colummns for expected data types 
df.info()
# Expected types for each column
expected_types = {'Transaction_ID': int,'tmsp': 'datetime64[ns]', 'country': object,'amount': int, 
                  'success': int, 'PSP': object, '3D_secured': int, 'card':object }

#Checking the columns to conform to the expected types
for col, expected_dtype in expected_types.items():
    actual_dtype = df[col].dtype
    if actual_dtype == expected_dtype:
        print(f"{col} matches the expected type: {expected_dtype}")
    else:
        print(f"{col} does NOT match the expected type. Expected: {expected_dtype}, but got: {actual_dtype}")

#Converting Pandas timestamp into Python datetime
df['tmsp'] = pd.to_datetime(df['tmsp'])
#Cheching the columns for negative values
columns_to_check = ['Transaction_ID','amount','success', '3D_secured']
negative_values = df[columns_to_check].apply(lambda x: x < 0)
negative_count = negative_values.sum()
print(f"Negative Values Count: {negative_count}")


#Presenting outliers with a boxplot and a scatter plot
import matplotlib.pyplot as plt

#Creating a box plot
plt.boxplot(df['amount'])
plt.title('Boxplot for Outlier Detection')
plt.show()

#Creating a scatter plot
plt.scatter(df.index, df['amount'])
plt.title('Scatter Plot for Outlier Detection')
plt.show()

#Checking the outliers with Z-score
from scipy import stats
import numpy as np

#Z-score calculation
df['zscore'] = np.abs(stats.zscore(df['amount']))
outliers = df[df['zscore'] > 3] #Setting a threshold for Z-score
print(outliers)
df = df.drop(columns=['zscore']) # no more needed

#Removing duplicates
#Sorting by customer, amount, country, and timestamp
df = df.sort_values(by=['Transaction_ID', 'amount', 'country', 'tmsp'])
#Creating the 'is_retry' flag and the retry sequence
df['is_retry'] = df.duplicated(subset=['amount', 'country'], keep=False) & (df['tmsp'].diff().dt.total_seconds() <= 60)

#Ranking the transactions to get retry sequence number
df['retry_count'] = df.groupby(['amount', 'country'])['tmsp'].rank(method='dense')

#Removing the retry duplicates and creating a new database while keeping the first occurence of each transaction
df_new = df[df['retry_count'] == 1].copy()
print(df)
print(df_new.shape)
print(df_new.head())


#Checking the original and cleaned dataset sizes
print(f"Original data size: {df.shape}")
print(f"Cleaned data size: {df_new.shape}")
