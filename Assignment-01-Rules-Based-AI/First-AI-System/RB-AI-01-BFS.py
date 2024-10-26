import pandas as pd 

# Import data file
df = pd.read_csv('./data/StudentPerformanceFactors.csv')

# Debug: ensure the program can reach the datafile 
# print(df)

# Basic Data Analysis:

# The Average Score
# Students who fall below this may be areas of concern
# But consider Skewed Data Distribution (high scores raising the mean)
averageScore = df["Exam_Score"].mean()

# The Median Score (value in the middle)
# Students who fall below this are in the lower 50% of the class
medianScore = df["Exam_Score"].median()

# The Mode Score (most common value)
# Good indicator to see students scoring lower than the most common value, possibly underperforming
# [0] on the end is only to get the first mode to prevent multiple values from being returned
modeScore = df["Exam_Score"].mode()[0]

print("The Mean (average) score is: " + str(averageScore))
print("The Median of the score is: " + str(medianScore))
print("The Mode score is: " + str(modeScore))


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


## ChatGPT 4o: I have a function to assign grades in a new column, how do I add a new column using pandas to apply to each exam score?
# Apply the grading function to each student's score
df['Grade'] = df['Exam_Score'].apply(assume_grade)

# Check the distribution of grades
print(df['Grade'].value_counts())
