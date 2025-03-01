'''Feature engineering: new features are created.
PSP_fee_success
PSP_fee_failure
PSP_fee_applied
transaction_fee
PSP_success_adjusted'''

#Creating 'PSP_fee_success', 'PSP_fee_failure', and 'PSP_fee_applied'
PSP_fees_success = {
    'Moneycard': 5, 
    'Goldcard': 10, 
    'UK_Card': 3, 
    'Simplecard': 1}
PSP_fees_failure = {
    'Moneycard': 2, 
    'Goldcard': 5, 
    'UK_Card': 1, 
    'Simplecard': 0.5}
df_new['PSP_fee_success'] = df_new['PSP'].map(PSP_fees_success)
df_new['PSP_fee_failure'] = df_new['PSP'].map(PSP_fees_failure)
df_new['PSP_fee_applied'] = df_new.apply(
    lambda x: x['PSP_fee_success'] if x['success'] == 1 else x['PSP_fee_failure'], axis=1
)

#Creating 'transaction_fee': calculating the transaction fee based on the amount and the applied PSP fee
df_new['transaction_fee'] = df_new['amount'] * df_new['PSP_fee_applied']

#Creating 'PSP_success_adjusted':
#Calculating the original success rate for each PSP
df_new['PSP_success_rate'] = df_new.groupby('PSP')['success'].transform('mean')

#Applying a penalty for PSPs with higher retry counts
df_new['avg_retry_count'] = df_new.groupby('PSP')['retry_count'].transform('mean')
#Penalizing the success rate by dividing it by (average retry count + 1)
df_new['PSP_success_adjusted'] = df_new['PSP_success_rate'] / (df_new['avg_retry_count'] + 1)

#Converting categorical variables into numerical with LabelEncoder
le = LabelEncoder()
df_new['country'] = le.fit_transform(df_new['country'])
df_new['card'] = le.fit_transform(df_new['card'])
df_new['PSP'] = le.fit_transform(df_new['PSP'])

#Cleaning the new dataset from unnecessary features
df_new = df_new.drop(columns=['is_retry','tmsp','retry_count','avg_retry_count','PSP_success_rate'])
print(df_new)
