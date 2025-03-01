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
