"""
    Rule Based AI System 01
    This Rule Based AI System predicts student grades based using a much higher set of rules.

    Last Modified: 24/11/2024 @ 10:39pm
"""


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

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
def predict_grade_with_rules(row, df):
    hoursStudied = row['Hours_Studied']
    attendance = row['Attendance']
    parentalInvolvement = row['Parental_Involvement']
    accessToResources = row['Access_to_Resources']
    extracurricularActivities = row['Extracurricular_Activities']
    previousScores = row['Previous_Scores']
    motivationLevel = row['Motivation_Level']
    sleepHours = row['Sleep_Hours']
    tutoringSessions = row['Tutoring_Sessions']
    teacherQuality = row['Teacher_Quality']
    learningDifficulties = row['Learning_Disabilities']

    averageHoursStudied = df['Hours_Studied'].mean()
    averageAttendance = df['Attendance'].mean()
    averagePreviousScores = df['Previous_Scores'].mean()
    averageSleepHours = df['Sleep_Hours'].mean()
    averageTutoringSessions = df['Tutoring_Sessions'].mean()
    
    # Initialize a score for grading
    grade_points = 0
    
    # Define rules based on previous scores
    if previousScores >= averagePreviousScores * 1.3:
        grade_points += 3 
    elif previousScores >= averagePreviousScores:
        grade_points += 2  
    elif previousScores >= averagePreviousScores * 0.6:
        grade_points += 1  
    
    # Rule based on access to resources
    if accessToResources == "High":
        grade_points += 2
    elif accessToResources == "Medium":
        grade_points += 1
    elif accessToResources == "Low":
        grade_points += 0
    
    # Rule based on extracurricular activities
    if extracurricularActivities == "Yes":
        grade_points += 0
    else:
        grade_points += 0
    
    # Rule based on motivation level
    if motivationLevel == "High":
        grade_points += 2
    elif motivationLevel == "Medium":
        grade_points += 2
    elif motivationLevel == "Low":
        grade_points += 1

    # Rule based on hours studied
    if hoursStudied >= averageHoursStudied * 1.3:  
        grade_points += 4 
    elif hoursStudied >= averageHoursStudied * 0.9:
        grade_points += 2  
    elif hoursStudied >= averageHoursStudied * 0.5:
        grade_points += 1
    else:
        grade_points -= 0
    
    # Rule based on attendance
    if attendance >= averageAttendance * 1.0:
        grade_points += 2 
    elif attendance >= averageAttendance * 0.8:
        grade_points += 0  
    
    # Rule based on parental involvement
    if parentalInvolvement == "High":
        grade_points += 2
    elif parentalInvolvement == "Medium":
        grade_points += 1  
    elif parentalInvolvement == "Low":
        grade_points += 1
    
    # Rule based on sleep hours
    if sleepHours <= averageSleepHours * 0.5:
        grade_points -= 2

    # Rule based on tutoring sessions
    if tutoringSessions >= averageTutoringSessions:
        grade_points += 1

    # Rule based on teacher quality
    if teacherQuality == "High":
        grade_points += 1
    elif teacherQuality == "Medium":
        grade_points += 1
    elif teacherQuality == "Low":
        grade_points += 0

    # Rule based on learning difficulties
    if learningDifficulties == "Yes":
        grade_points -= 1
    elif learningDifficulties == "No":
        grade_points += 0
    
    # Assign grade based on total grade points
    if grade_points >= 15:
        return 'A'
    elif grade_points >= 13:
        return 'B'
    elif grade_points >= 8:
        return 'C'
    elif grade_points >= 6:
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
    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    # Determine the actual and predicted grades
    df['Actual_Grade'] = df['Exam_Score'].apply(assume_grade)
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade_with_rules(row, df), axis=1)

    # Define the set of all possible grades
    grade_categories = ['A', 'B', 'C', 'D', 'F']

    ## ChatGPT 4o: Rules based AI how do you assign calculations using a confusion matrix for multiple grades python 
    # This code which has been generated by ChatGPT creates a table for the confusion matrix, actual (assumed) against predicted
    confusion_matrix = pd.crosstab(df['Predicted_Grade'], df['Actual_Grade'], rownames=['Predicted'], colnames=['Actual'], dropna=False)
    confusion_matrix = confusion_matrix.reindex(index=grade_categories, columns=grade_categories, fill_value=0)

    # Calculate and print additional metrics
    precision = precision_score(df['Actual_Grade'], df['Predicted_Grade'], average='macro')
    recall = recall_score(df['Actual_Grade'], df['Predicted_Grade'], average='macro')
    f1 = f1_score(df['Actual_Grade'], df['Predicted_Grade'], average='macro')
    accuracy = accuracy_score(df['Actual_Grade'], df['Predicted_Grade'])

    # Calculate accuracy using maths
    correct_predictions = sum(confusion_matrix.iloc[i, i] for i in range(len(grade_categories)))
    total_predictions = confusion_matrix.values.sum()
    accuracy = correct_predictions / total_predictions

    # Output the Confusion Matrix for analystics purposes
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Print Metrics to console
    print("\nEvaluation Metrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    # This code is generated by ChatGPT, please see above comment regarding ChatGPT 4o usage.
    # Copy the original matrix and add an additional row for column totals
    confusion_matrix_with_totals = confusion_matrix.copy()  
    confusion_matrix_with_totals.loc['Total'] = confusion_matrix.sum(axis=0)  
    updated_labels = grade_categories + ['Actual Total']

    # Plot and display to the program operator
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_with_totals, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=grade_categories, yticklabels=updated_labels)
    plt.xlabel("Actual (Assumed) Grades")
    plt.ylabel("Predicted Grades")
    plt.title("Confusion Matrix for Grade Prediction with Totals")
    plt.show()


# Run the main function
main()
