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
