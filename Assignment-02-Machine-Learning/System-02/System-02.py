# Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/predictive_maintaince_compressed.csv')

# Prepare data
X = df.drop(columns=['datetime', 'Failure']) 
y = df['Failure']  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Reference (ChatGPT 4o) How can I use Support Vector Machine algorithm for a machine learning model?
# Train the Random Forest classifier
clf = SVC(probability=True, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class

# Get the accuracy, precision, recall, f1 score, log loss and PR-AUC
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
log_loss_val = log_loss(y_test, y_prob)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_vals, precision_vals)

# Store metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Log Loss": log_loss_val,
    "PR-AUC": pr_auc
}

# Output metrics to console
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# -------------------------------
#          Graphs
# -------------------------------

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
