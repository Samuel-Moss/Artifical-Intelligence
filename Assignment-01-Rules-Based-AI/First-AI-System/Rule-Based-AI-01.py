"""
    Rule Based AI System 01
    This Rule Based AI System predicts student grades.

    Last Modified: 05/11/2024 @ 12:34pm
"""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ------------------------------------------------------------
#                      Grade Assumptions
# ------------------------------------------------------------
#  Assume student grades due to missing data within the dataset
# 
#  The assumed grade is to be considered the 'actual' grade  

# The data set has no indication of grades, so these are to fulfill the missing data
def assume_grade(score):

    # If else statements
    if score >= 76:
        return 'A'
    elif score >= 71:
        return 'B'
    elif score >= 65:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'
    

# ------------------------------------------------------------
#                      Grade Predictions
# ------------------------------------------------------------
#  Sum up numbered factors and compare against the 
#  student average to determine grade
# 
#  The assumed grade is to be considered the 'actual' grade.  

def predict_grade(row, df):
    # Summing up the specified factors that do the following:
        # 1 - Do scientifically contribute to student grades
        # 2 - AND is a number value - needed to do calculations
    score_sum = (
        row['Hours_Studied'] +
        row['Attendance'] +
        row['Previous_Scores'] +
        row['Sleep_Hours'] +
        row['Tutoring_Sessions']
    )

    # Calculate the average of the specified factors for comparison
    average_sum = (
        df['Hours_Studied'].mean() +
        df['Attendance'].mean() +
        df['Previous_Scores'].mean() +
        df['Sleep_Hours'].mean() +
        df['Tutoring_Sessions'].mean()
    )

    # Assign predicted grade based on the sum compared to the average

    # If above average (1.25x above average)
    if score_sum >= average_sum * 1.25:
        return 'A'
    
    # Else if around average (between 1x - 1.25x average)
    elif score_sum >= average_sum * 1:
        return 'B'

    # Else if just below average (between 0.85x - 1x average)
    elif score_sum >= average_sum * 0.85:
        return 'C'
    
    # Else if further below average (between 0.75 - 0.85x average)
    elif score_sum >= average_sum * 0.75:
        return 'D'
    
    # Else severly below average
    else:
        return 'F'

# -------------------------------
#          Main Function
# -------------------------------
#  Entry point 
#
#  References:
#  OpenAI ChatGPT 4o (2023) ChatGPT response to myself, 5 November.
#  (A copy of prompts can be found below)

def main():

    # Define the grade categories to ensure consistent order (this needs to be used within confusion matrix)
    grade_categories = ['A', 'B', 'C', 'D', 'F']

    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    # Apply assumed ("actual") & predicted grade
    df['Assumed_Grade'] = df['Exam_Score'].apply(assume_grade)
    ## ChatGPT 4o: I have a function to predict grades, how do I apply it to my dataframe? (python rules based AI)
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade(row, df), axis=1)

    # Print the distribution of actual grades to the console
    print("Actual Grade Distribution:\n", df['Assumed_Grade'].value_counts())

    ## ChatGPT 4o: How do you generate a confusion matrix comparing two 'Predicted_Grade' and 'Assumed_Grade' from a pandas df (python rule based AI), it needs to be used to calculate accuracy (correct predictions / the total num of predictions)
    # Calculate confusion matrix
    confusion_matrix = pd.crosstab(df['Predicted_Grade'], df['Assumed_Grade'], rownames=['Predicted'], colnames=['Assumed'], dropna=False)
    confusion_matrix = confusion_matrix.reindex(index=grade_categories, columns=grade_categories, fill_value=0)

    # Calculate correct and total predictions
    correct_predictions = sum(confusion_matrix.iloc[i, i] for i in range(len(grade_categories)))
    total_predictions = confusion_matrix.values.sum()

    # Calculate and print additional metrics
    precision = precision_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    recall = recall_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    f1 = f1_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    accuracy = accuracy_score(df['Assumed_Grade'], df['Predicted_Grade'])

    # Print the confusion matrix to console
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Print Metrics to console
    print("\nEvaluation Metrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)



    """
    # Report Diagrams

    
    # Plot confusion Matrix:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=grade_categories, yticklabels=grade_categories)
    plt.xlabel("Actual Grades")
    plt.ylabel("Predicted Grades")
    plt.title("Confusion Matrix for Grade Prediction")
    plt.show()

    
    
    """


    # Print data types
    # print("\nData Types Summary:")
    # print(df.dtypes)

# Run the entry point
main()
