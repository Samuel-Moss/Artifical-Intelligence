import pandas as pd 

# The data set has no indication of grades, so these are to fulfil the missing data
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

    # Some of these are unused!
    score = row['Exam_Score']
    hoursStudied = row['Hours_Studied']
    attendance = row['Attendance']
    parentalInvolement = row['Parental_Involvement']
    accessToResources = row['Access_to_Resources']
    preiousScores = row['Previous_Scores']
    motivationLevel = row['Motivation_Level']
    sleepHours = row['Sleep_Hours']

    # Get Average values
    averageScore = df["Exam_Score"].mean()
    averageHoursStudied = df["Hours_Studied"].mean()

        # Attendance
            # If attendance is 1.5x the average, predict A
            # If attendance is 1.2x the average, predict B
    
    # If Else stements to predict grades
    if hoursStudied >= averageHoursStudied * 1.5 and attendance >= 90:
        return 'A'
    elif hoursStudied >= averageHoursStudied * 1.2 and attendance >= 80:
        return 'B'
    elif hoursStudied >= averageHoursStudied and attendance >= 70:
        return 'C'
    elif hoursStudied >= averageHoursStudied * 0.8 and attendance >= 60:
        return 'D'
    else:
        return 'F'

def main():
    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    ## ChatGPT 4o: I have a function to assign grades in a new column, how do I add a new column using pandas to apply to each exam score?
    # Apply the grading function to each student's score
    df['Actual_Grade'] = df['Exam_Score'].apply(assume_grade)

    # Print the distribution of grades to the console
    print(df['Actual_Grade'].value_counts())

    ## ChatGPT 4o: I have a function to predict grades, how do I apply it to my dataframe? (python rules based AI)
    # Apply the prediction function to the DataFrame
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade(row, df), axis=1)

    # Print the predicted grades based on attendance
    print(df[['Hours_Studied', 'Attendance', 'Predicted_Grade', 'Actual_Grade']])

    # Print the distribution of predicted grades against their actual grade to console
    print(df[['Predicted_Grade', 'Actual_Grade']].value_counts())

# Run entry point
main()
