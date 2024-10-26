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
