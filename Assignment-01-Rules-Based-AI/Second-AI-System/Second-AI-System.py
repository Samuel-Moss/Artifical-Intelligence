import pandas as pd 

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

def predict_grade_with_rules(row):
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
    
    # Initialize a score for grading
    grade_points = 0
    
    # Define rules based on previous scores
    if previousScores >= 75:
        grade_points += 3  
    elif previousScores >= 65:
        grade_points += 2  
    elif previousScores >= 50:
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
        grade_points += 2
    else:
        grade_points += 0
    
    # Rule based on motivation level
    if motivationLevel == "High":
        grade_points += 2
    elif motivationLevel == "Medium":
        grade_points += 1
    elif motivationLevel == "Low":
        grade_points += 0

    # Rule based on hours studied
    if hoursStudied >= 30:  
        grade_points += 4  
    elif hoursStudied >= 20:
        grade_points += 2  
    elif hoursStudied >= 10:
        grade_points += 1
    else:
        grade_points -= 1
    
    # Rule based on attendance
    if attendance >= 90:
        grade_points += 2 
    elif attendance >= 75:
        grade_points += 0.5  
    
    # Rule based on parental involvement
    if parentalInvolvement == "High":
        grade_points += 2
    elif parentalInvolvement == "Medium":
        grade_points += 1  
    elif parentalInvolvement == "Low":
        grade_points += 0
    
    # Rule based on sleep hours
    if sleepHours <= 6:
        grade_points -= 0 
    
    # Rule based on tutoring sessions
    if tutoringSessions >= 2:
        grade_points += 1  # Add points for having tutoring sessions

    # Rule based on teacher quality
    if teacherQuality == "High":
        grade_points += 2
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
    if grade_points >= 18:
        return 'A'
    elif grade_points >= 13:
        return 'B'
    elif grade_points >= 8:
        return 'C'
    elif grade_points >= 6:
        return 'D'
    else:
        return 'F'

def main():
    # Import data file
    df = pd.read_csv('./data/StudentPerformanceFactors.csv')

    df['Actual_Grade'] = df['Exam_Score'].apply(assume_grade)

    # Apply the rule-based prediction function to the DataFrame
    df['Predicted_Grade'] = df.apply(predict_grade_with_rules, axis=1)

    # Print the distribution of predicted grades
    print(df[['Predicted_Grade', 'Actual_Grade']].value_counts())


    # Calculate and print accuracy
    correct_predictions = (df['Predicted_Grade'] == df['Actual_Grade']).sum()
    accuracy = correct_predictions / len(df)

    print("Accuracy of the grade prediction:", accuracy)

# Run entry point
main()