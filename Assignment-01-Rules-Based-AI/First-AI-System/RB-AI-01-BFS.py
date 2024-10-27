import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The data set has no indication of grades, so these are to fulfill the missing data
def assume_grade(score):
    if score >= 75:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 65:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def predict_grade(row, df):
    # Summing up the specified factors
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
    if score_sum >= average_sum * 1.2:
        return 'A'
    elif score_sum >= average_sum * 1:
        return 'B'
    elif score_sum >= average_sum * 0.8:
        return 'C'
    elif score_sum < average_sum * 0.8:
        return 'D'
    else:
        return 'F'

def main():
    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    # Apply the grading function to each student's score for actual grades
    df['Actual_Grade'] = df['Exam_Score'].apply(assume_grade)

    # Print the distribution of actual grades to the console
    print("Actual Grade Distribution:\n", df['Actual_Grade'].value_counts())

    # Apply the updated prediction function to the DataFrame
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade(row, df), axis=1)

    # Define the grade categories to ensure consistent order in the confusion matrix
    grade_categories = ['A', 'B', 'C', 'D', 'F']

    # Calculate confusion matrix
    confusion_matrix = pd.crosstab(df['Predicted_Grade'], df['Actual_Grade'], rownames=['Predicted'], colnames=['Actual'], dropna=False)
    confusion_matrix = confusion_matrix.reindex(index=grade_categories, columns=grade_categories, fill_value=0)


    correct_predictions = sum(confusion_matrix.iloc[i, i] for i in range(len(grade_categories)))
    total_predictions = confusion_matrix.values.sum()
    accuracy = correct_predictions / total_predictions

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("\nAccuracy of the grade prediction:", accuracy)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=grade_categories, yticklabels=grade_categories)
    plt.xlabel("Actual Grades")
    plt.ylabel("Predicted Grades")
    plt.title("Confusion Matrix for Grade Prediction")
    plt.show()

# Run the entry point
main()
