import pandas as pd 

def assume_grade(score):
    if score >= 80:
        return 'A'
    elif score >= 71:
        return 'B'
    elif score >= 63:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

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
        grade_points += 1
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
        grade_points += 3  
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
        grade_points -= 0
    elif learningDifficulties == "No":
        grade_points += 0
    
    # Assign grade based on total grade points
    if grade_points >= 19:
        return 'A'
    elif grade_points >= 13:
        return 'B'
    elif grade_points >= 8:
        return 'C'
    elif grade_points >= 5:
        return 'D'
    else:
        return 'F'

def main():
    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    df['Actual_Grade'] = df['Exam_Score'].apply(assume_grade)

    # Apply the rule-based prediction function to the DataFrame
    df['Predicted_Grade'] = df.apply(lambda row: predict_grade_with_rules(row, df), axis=1)

    # Print the distribution of predicted grades
    print(df[['Predicted_Grade', 'Actual_Grade']].value_counts())


    # Calculate and print accuracy
    correct_predictions = (df['Predicted_Grade'] == df['Actual_Grade']).sum()
    accuracy = correct_predictions / len(df)

    print("Accuracy of the grade prediction:", accuracy)

# Run entry point
main()
