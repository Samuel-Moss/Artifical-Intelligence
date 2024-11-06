"""
    Rule Based AI System 01
    This Rule Based AI System predicts student grades.

    Last Modified: 05/11/2024 @ 12:34pm
"""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tracemalloc
import time

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ------------------------------------------------------------
#                      Grade Assumptions
# ------------------------------------------------------------
# Assume student grades due to missing data within the dataset

def assume_grade(score):
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
def predict_grade(row, df):
    score_sum = (
        row['Hours_Studied'] +
        row['Attendance'] +
        row['Previous_Scores'] +
        row['Sleep_Hours'] +
        row['Tutoring_Sessions']
    )

    average_sum = (
        df['Hours_Studied'].mean() +
        df['Attendance'].mean() +
        df['Previous_Scores'].mean() +
        df['Sleep_Hours'].mean() +
        df['Tutoring_Sessions'].mean()
    )

    if score_sum >= average_sum * 1.25:
        return 'A'
    elif score_sum >= average_sum * 1:
        return 'B'
    elif score_sum >= average_sum * 0.85:
        return 'C'
    elif score_sum >= average_sum * 0.75:
        return 'D'
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

    # Start tracemalloc for memory tracking and time tracking
    tracemalloc.start()
    start_time = time.time()

    # Define grade categories to ensure consistent order
    grade_categories = ['A', 'B', 'C', 'D', 'F']

    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    # Apply assumed ("actual") & predicted grades
    df['Assumed_Grade'] = df['Exam_Score'].apply(assume_grade)
    ## ChatGPT 4o: I have a function to predict grades, how do I apply it to my dataframe? (python rules based AI)
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade(row, df), axis=1)

    # Print the distribution of actual grades to the console
    print("Actual Grade Distribution:\n", df['Assumed_Grade'].value_counts())

    ## ChatGPT 4o: How do you generate a confusion matrix comparing two 'Predicted_Grade' and 'Assumed_Grade' from a pandas df (python rule based AI), it needs to be used to calculate accuracy (correct predictions / the total num of predictions)
    # Calculate confusion matrix
    confusion_matrix = pd.crosstab(df['Predicted_Grade'], df['Assumed_Grade'], rownames=['Predicted'], colnames=['Assumed'], dropna=False)
    confusion_matrix = confusion_matrix.reindex(index=grade_categories, columns=grade_categories, fill_value=0)

    # Adding only the 'Actual Total' row to the confusion matrix
    confusion_matrix.loc['Actual Total'] = confusion_matrix.sum(axis=0)  # Column totals

    # Print the updated confusion matrix with only the total row
    print("Confusion Matrix with Grade Distributions:")
    print(confusion_matrix)

    # Calculate correct and total predictions
    correct_predictions = sum(confusion_matrix.iloc[i, i] for i in range(len(grade_categories)))
    total_predictions = confusion_matrix.values.sum()

    # Calculate and print additional metrics
    precision = precision_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    recall = recall_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    f1 = f1_score(df['Assumed_Grade'], df['Predicted_Grade'], average='macro')
    accuracy = accuracy_score(df['Assumed_Grade'], df['Predicted_Grade'])

    # Print Metrics to console
    print("\nEvaluation Metrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    ## ChatGPT 4o: How do I calculate execution time and memory usage in my python program?
    # End the time and the memory tracking
    execution_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print hardware metrics
    print("\nHardware Metrics:")
    print("Execution Time:", execution_time, "seconds")
    print("Current Memory Usage:", current_memory / 1024, "KB")
    print("Peak Memory Usage:", peak_memory / 1024, "KB")

    # Plot confusion Matrix with only the 'Actual Total' row
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=list(confusion_matrix.columns), yticklabels=list(confusion_matrix.index))
    plt.xlabel("Actual (Assumed) Grades")
    plt.ylabel("Predicted Grades")
    plt.title("SYSTEM 01: Confusion Matrix for Grade Prediction with Distributions")
    plt.show()

    # Print data types
    # print("\nData Types Summary:")
    # print(df.dtypes)

# Run the main function
main()