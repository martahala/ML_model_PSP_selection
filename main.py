'''Since the project is more about experiment, the code is not structured with well-defined classes and functions.
'''
''' Data understanding and preparation '''
#Uploading the given dataset using Pandas 
import pandas as pd
df = pd.read_excel('/Users/marththe/Desktop/Model Engineering/PSP_Jan_Feb_2019.xlsx')

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


'''Feature selection with techniques: 
correlation matrix
'''

#Correlation matrix
corr_matrix = df_new.corr()
print(corr_matrix)

#Getting correlations between all features and the target variable
target_correlation = corr_matrix['success'].sort_values(ascending=False)

#Printing the outcomes
print("Top positively correlated features with target:")
print(target_correlation[target_correlation > 0.1])  # Threshold of 0.1 for positive correlation

print("\nTop negatively correlated features with target:")
print(target_correlation[target_correlation < -0.1])  # Threshold of -0.1 for negative correlation

#Visualizing the correlation matrix
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show

'''Feature selection with techniques: 
RandomForestClassifier
'''
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
X = df_new[['PSP_fee_applied', 'transaction_fee','country','3D_secured','card','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure' ]]
y = df_new['success']

#Initializing the model
model = RandomForestClassifier()

#Applying RFECV for feature elimination with cross-validation
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X, y)

#Printing the outcome
print("Optimal number of features:", rfecv.n_features_)


'''Feature selection with techniques: 
Benjamini-Hochberg FDR
'''
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
np.random.seed(42)

#Splitting the data based on the target
group1 = df_new[df_new['success'] == 0]
group2 = df_new[df_new['success'] == 1]

#Performing t-tests
p_values = []
for feature in ['PSP_fee_applied', 'transaction_fee','country','3D_secured','card','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure' ]:
    t_stat, p_val = ttest_ind(group1[feature], group2[feature])
    p_values.append(p_val)

#Converting p_values to a numpy array
p_values = np.array(p_values)

#Applying Benjamini-Hochberg FDR control
alpha = 0.05  #FDR threshold
rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

#Creating a result DataFrame
results = pd.DataFrame({
    'Feature': [ 'PSP_fee_applied', 'transaction_fee','country','3D_secured','card','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure' ],
    'P-value': p_values,
    'Corrected P-value': pvals_corrected,
    'Rejected (Significant)': rejected
})

print(results)

'''Feature selection with techniques: 
Chi-squared
'''
from sklearn.feature_selection import SelectKBest, chi2
X = df_new[['PSP_fee_applied', 'transaction_fee','country','3D_secured','card','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure' ]]
y = df_new['success']

#Performing Chi-Squared test
selector = SelectKBest(chi2, k=4)  #Selecting 4 features
X_new = selector.fit_transform(X, y)

#Getting the chi-squared scores and p-values
chi2_scores = selector.scores_
p_values = selector.pvalues_

#Showing the selected features and p-values
print("Selected Features:\n", X.columns[selector.get_support()])
print("P-values of Features:\n", p_values)
#Creating a DataFrame for visualization
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'P-value': p_values
}).sort_values(by='Chi2 Score', ascending=False)

#Bar plot for Chi2 scores
plt.figure(figsize=(8, 6))
plt.barh(feature_scores['Feature'], feature_scores['Chi2 Score'], color='skyblue')
plt.xlabel('Chi2 Score')
plt.ylabel('Feature')
plt.title('Chi-Squared Feature Importance')
plt.gca().invert_yaxis()
plt.show()
'''Feature selection with techniques: 
RandomForestClassifier
'''
X = df_new[['PSP_fee_applied', 'transaction_fee','country','3D_secured','card','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure' ]]
y = df_new['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
#Fitting the model
model.fit(X, y)

#Getting feature importances
importances = model.feature_importances_

#Ranking features by importance
important_features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(important_features)

'''Feature selection with techniques: 
ANOVA F-Test
'''
from sklearn.feature_selection import SelectKBest, f_classif

#Applying SelectKBest
f_classif_selector = SelectKBest(score_func=f_classif, k=4)  # Select top 2 features
X_kbest_f_classif = f_classif_selector.fit_transform(X, y)

#Displaying selected feature names and their scores
selected_features_f_classif = X.columns[f_classif_selector.get_support()]
print("Selected features using ANOVA F-Test:", selected_features_f_classif)
print("ANOVA F-Test Scores:", f_classif_selector.scores_)  

#Visualizing the results
features = X.columns
scores = f_classif_selector.scores_

# Sort scores and features in descending order for better visualization
sorted_indices = np.argsort(scores)[::-1]
sorted_scores = scores[sorted_indices]
sorted_features = features[sorted_indices]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_scores, color='skyblue')
plt.xlabel('ANOVA F-Test Score')
plt.ylabel('Features')
plt.title('Feature Importance based on ANOVA F-Test')
plt.gca().invert_yaxis()  
plt.show()

'''Feature selection with techniques: 
L1 Norm
'''
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
#Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Applying Lasso with L1 regularization
lasso = Lasso(alpha=0.01)  
lasso.fit(X_scaled, y)

#Getting the coefficients
coefficients = lasso.coef_

#Creating a DataFrame to display the features 
feature_importance = pd.DataFrame({
    'Feature': X.columns,  # If X is a DataFrame, otherwise use a list of feature names
    'Coefficient': coefficients
})

#Filtering out features with zero coefficients 
selected_features = feature_importance[feature_importance['Coefficient'] != 0]
print("Selected features:")
print(selected_features)

#Visualizing the outcome
plt.figure(figsize=(10, 6))
plt.barh(selected_features['Feature'], selected_features['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Lasso Regression Feature Importance (L1 Norm)')
plt.show()


'''Building a machine learning model:
Analyze '3D_secured' feature influence
'''
#Filtering the dataset for secured transactions
secured_transactions = df_new[df_new['3D_secured'] == 1]

#Grouping by PSP to calculate the success rate of secured transactions
secured_psp_performance = secured_transactions.groupby('PSP').agg({
    'PSP_success_adjusted': 'mean',           # Calculate success rate
    'amount': 'mean',            # Average amount for secured transactions
    'Transaction_ID': 'count'    # Count of secured transactions
}).reset_index()

#Renaming the columns for clarity
secured_psp_performance.columns = ['PSP', 'secured_success_rate', 'avg_secured_amount', 'secured_transaction_count']


#Calculating success rates for secured and unsecured transactions
secured_vs_unsecured = df_new.groupby('3D_secured').agg({
    'PSP_success_adjusted': 'mean',           # Success rate
    'amount': 'mean',            # Average amount
    'Transaction_ID': 'count'    # Count of transactions
}).reset_index()

#Renaming the columns
secured_vs_unsecured.columns = ['3D_secured', 'success_rate', 'avg_amount', 'transaction_count']

print(secured_vs_unsecured)
print(secured_psp_performance)

'''Building a machine learning model:
A logistic regression model (baseline)
'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

features = ['PSP_fee_success', 'PSP_fee_failure','3D_secured','amount','PSP','card','country']
X = df_new[features]
y = df_new['success']

#Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the logistic regression model
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train, y_train)

#Predicting the probability of success for each transaction
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

#Converting the predicted probabilities into binary predictions
X_test['final_prediction'] = (y_pred_prob > 0.5).astype(int)

#Evaluating the baseline model's accuracy
accuracy = accuracy_score(y_test, X_test['final_prediction'])
precision = precision_score(y_test, X_test['final_prediction'])
recall = recall_score(y_test, X_test['final_prediction'])
f1 = f1_score(y_test, X_test['final_prediction'])
roc_auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, X_test['final_prediction'])
#Printing the metrics
print(f"Baseline Logistic Regression Model Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

#Displaying confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

#Displaying the first predictions
print(X_test[['amount', '3D_secured','PSP', 'final_prediction']].head())


'''Building a machine learning model:
A random forest model (an accurate model)
Applying the outcomes of the '3D_secured' feature
'''
features=['transaction_fee','3D_secured','amount','PSP','Transaction_ID','PSP_success_adjusted','PSP_fee_success','PSP_fee_failure']

X = df_new[features]
y = df_new['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

#Predicting the probability of success for each transaction
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

#Converting the predicted probabilities into binary predictions (final success prediction)
threshold = 0.3
X_test['final_prediction'] = (y_pred_prob > threshold).astype(int)

#Defining the success rates and fees for each PSP
psp_secured_success_rates = {
    0: 0.191489,  # PSP 0 secured success rate
    1: 0.109914,  # PSP 1 secured success rate
    2: 0.069014,  # PSP 2 secured success rate
    3: 0.099057   # PSP 3 secured success rate
}
#Weights for balancing success rate and fees
alpha = 0.7  # More weight to success rate
beta = 0.3   # Less weight to minimizing fees
#Function to calculate the cost-adjusted score for PSPs
def calculate_psp_score(row, psp_success_rate):
    #Calculating the expected fee (weighted between success and failure fees)
    expected_fee = row['PSP_fee_success'] * row['final_prediction'] + row['PSP_fee_failure'] * (1 - row['final_prediction'])
    
    #Calculate the score: prioritizing success rate, penalizing high fees
    score = alpha * psp_success_rate[row['PSP']] - beta * expected_fee
    return score

#Applying the score calculation for each row based on the PSP
X_test['psp_score'] = X_test.apply(lambda row: calculate_psp_score(row, psp_secured_success_rates), axis=1)

#Defining the logic to choose the best PSP based on 3D-secured status and cost-based scores
def choose_best_psp(row):
    if row['3D_secured'] == 1:
        #Prioritizing PSP 0 for secured transactions, fallback to PSP 3 for bulk or lower-priority transactions
        if row['amount'] <= 250 and psp_secured_success_rates[0] > 0.5:
            return 0  #PSP 0 for smaller secured transactions with high success rate
        elif row['psp_score'] > 0:
            return row['PSP']  #Keeping the current PSP if score is positive
        else:
            return 3  #Using PSP 3 for larger transactions or if no other option is better
    else:
        #For unsecured transactions, keeping the original PSP selection
        return row['PSP']

#Applying the PSP decision logic for each transaction
X_test['chosen_psp'] = X_test.apply(choose_best_psp, axis=1)
#Evaluating the model's performance after applying PSP logic
accuracy = accuracy_score(y_test, X_test['final_prediction'])
print(f"Random Forest Accuracy: {accuracy}")

#Displaying the chosen PSP for each transaction
print(X_test[['Transaction_ID', 'amount', '3D_secured', 'chosen_psp', 'final_prediction']].head())
#Analyzing the distribution of chosen PSPs for secured transactions
secured_transactions = X_test[X_test['3D_secured'] == 1]
psp_choice_counts = secured_transactions['chosen_psp'].value_counts()
print("Chosen PSP distribution for secured transactions:")
print(psp_choice_counts)

#Evaluating the model's accuracy
accuracy = accuracy_score(y_test, X_test['final_prediction'])
precision = precision_score(y_test, X_test['final_prediction'])
recall = recall_score(y_test, X_test['final_prediction'])
f1 = f1_score(y_test, X_test['final_prediction'])
roc_auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, X_test['final_prediction'])

#Print the metrics
print(f"Baseline Logistic Regression Model Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

#Displaying confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

#Error analysis with SHAP
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

''' Checking the model's outcomes. Creating a new dataset with the outcomes. Cheching the errors'''

#Adding the chosen PSP to the historical data
X_test['historical_psp'] = X_test['PSP']  # Historical PSP column
X_test['chosen_psp'] = X_test.apply(choose_best_psp, axis=1)

#Creating the evaluation table with all relevant columns
evaluation_table = X_test[['Transaction_ID', 'amount', '3D_secured', 'historical_psp', 'chosen_psp', 'final_prediction']]

#Displaying the first few rows of the evaluation table
print(evaluation_table.head())

#Evaluating the model's performance by comparing the final prediction to the true labels
accuracy = accuracy_score(y_test, X_test['final_prediction'])

#Saving the evaluation table to a CSV for further analysis (optional)
evaluation_table.to_csv('psp_evaluation_table.csv', index=False)

#Checking the errors
errors = evaluation_table[evaluation_table['historical_psp'] != evaluation_table['chosen_psp']]
print(errors)

#Overall success of the model
overall_success_rate_historical = evaluation_table['final_prediction'][evaluation_table['historical_psp'] == evaluation_table['chosen_psp']].mean()
overall_success_rate_chosen = evaluation_table['final_prediction'][evaluation_table['historical_psp'] != evaluation_table['chosen_psp']].mean()

print(f"Overall success rate for historical PSP: {overall_success_rate_historical}")
print(f"Overall success rate for chosen PSP: {overall_success_rate_chosen}")
