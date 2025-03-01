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
