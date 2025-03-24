import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

def loading_screen():
    # Customize the loading bar with different colors and symbols
    bar_len = 30
    bar_symbol = "█"
    empty_symbol = "-"
    bar_color = "\033[1;32;40m"  # Green background with black text
    empty_color = "\033[1;37;40m"  # White background with black text

    # Clear the screen
    sys.stdout.write("\033[2J\033[H")

    # Show the loading bar
    for i in range(bar_len):
        time.sleep(0.1)
        progress = bar_symbol * i + empty_symbol * (bar_len - i)
        percent = (i / (bar_len - 1)) * 100
        sys.stdout.write("\r[{0}] {1:.0f}%".format(bar_color + progress + empty_color, percent))
        sys.stdout.flush()

    # Clear the loading bar
    sys.stdout.write("\r" + " " * (bar_len + 10) + "\r")
    sys.stdout.flush()

# Call the loading screen function
loading_screen()

# Your code goes here...
print("Activating Diabetes Prediction Model")


# Continue with the rest of your code

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Split the dataset into features and target variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)
svm = SVC(kernel='linear', probability=True)

# Create the ensemble model
ensemble = VotingClassifier(estimators=[('nb', nb), ('rf', rf), ('lr', lr)], voting='soft')

# Fit the models to the training data
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

# Make predictions on the test data
y_pred_nb = nb.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)

# Get prediction probabilities for ROC curve
y_prob_nb = nb.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_svm = svm.predict_proba(X_test)[:, 1]
y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]

# Evaluate the models' accuracy, F1 score, precision, recall, and AUC score
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_roc_auc = roc_auc_score(y_test, y_prob_nb)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_prob_rf)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_prob_lr)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_roc_auc = roc_auc_score(y_test, y_prob_svm)

ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_precision = precision_score(y_test, y_pred_ensemble)
ensemble_recall = recall_score(y_test, y_pred_ensemble)
ensemble_roc_auc = roc_auc_score(y_test, y_prob_ensemble)

#Print the evaluation metrics
print("Naive Bayes model's accuracy: {:.2f}%".format(nb_accuracy * 100))
print("Naive Bayes model's F1 score: {:.2f}%".format(nb_f1 * 100))
print("Naive Bayes model's precision: {:.2f}%".format(nb_precision * 100))
print("Naive Bayes model's recall: {:.2f}%".format(nb_recall * 100))
print("Naive Bayes model's ROC AUC score: {:.2f}%".format(nb_roc_auc * 100))

print("\nRandom Forest model's accuracy: {:.2f}%".format(rf_accuracy * 100))
print("Random Forest model's F1 score: {:.2f}%".format(rf_f1 * 100))
print("Random Forest model's precision: {:.2f}%".format(rf_precision * 100))
print("Random Forest model's recall: {:.2f}%".format(rf_recall * 100))
print("Random Forest model's ROC AUC score: {:.2f}%".format(rf_roc_auc * 100))

print("\nLogistic Regression model's accuracy: {:.2f}%".format(lr_accuracy * 100))
print("Logistic Regression model's F1 score: {:.2f}%".format(lr_f1 * 100))
print("Logistic Regression model's precision: {:.2f}%".format(lr_precision * 100))
print("Logistic Regression model's recall: {:.2f}%".format(lr_recall * 100))
print("Logistic Regression model's ROC AUC score: {:.2f}%".format(lr_roc_auc * 100))

print("\nSVM model's accuracy: {:.2f}%".format(svm_accuracy * 100))
print("SVM model's F1 score: {:.2f}%".format(svm_f1 * 100))
print("SVM model's precision: {:.2f}%".format(svm_precision * 100))
print("SVM model's recall: {:.2f}%".format(svm_recall * 100))
print("SVM model's ROC AUC score: {:.2f}%".format(svm_roc_auc * 100))

print("\nEnsemble model's accuracy: {:.2f}%".format(ensemble_accuracy * 100))
print("Ensemble model's F1 score: {:.2f}%".format(ensemble_f1 * 100))
print("Ensemble model's precision: {:.2f}%".format(ensemble_precision * 100))
print("Ensemble model's recall: {:.2f}%".format(ensemble_recall * 100))
print("Ensemble model's ROC AUC score: {:.2f}%".format(ensemble_roc_auc * 100))

# Generate confusion matrix for logistic regression model
lr_cm = confusion_matrix(y_test, y_pred_lr)

# Plot confusion matrix using seaborn heatmap
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Generate confusion matrix for Gaussian Naive Bayes
nb_cm = confusion_matrix(y_test, y_pred_nb)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
plt.title("Gaussian Naive Bayes Confusion Matrix")
sns.heatmap(nb_cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Generate the confusion matrix for SVM model
svm_cm = confusion_matrix(y_test, y_pred_svm)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(svm_cm, annot=True, cmap=plt.cm.Blues, fmt="d", cbar=False)
plt.title("Confusion Matrix for Support Vector Machine Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Get the confusion matrix for Random Forest model
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix using a heatmap
sns.heatmap(rf_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Get the confusion matrix for the ensemble model
y_pred_ensemble = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ensemble)

# Plot the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Ensemble Model (Soft Voting)')
plt.show()

#Plot ROC curves
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_prob_ensemble)

roc_auc_nb = auc(fpr_nb, tpr_nb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

#Plot ROC curves
plt.plot(fpr_nb, tpr_nb, label="Naive Bayes (AUC = {:.2f})".format(nb_roc_auc))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(rf_roc_auc))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = {:.2f})".format(lr_roc_auc))
plt.plot(fpr_svm, tpr_svm, label="SVM (AUC = {:.2f})".format(svm_roc_auc))
plt.plot(fpr_ensemble, tpr_ensemble, label="Ensemble (AUC = {:.2f})".format(ensemble_roc_auc))

#Set the title, labels, and legend
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()

#Show the plot
plt.show()

# Create a bar chart comparing the models' evaluation metrics
labels = ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Ensemble']
accuracy_scores = [nb_accuracy, rf_accuracy, lr_accuracy, svm_accuracy, ensemble_accuracy]
precision_scores = [nb_precision, rf_precision, lr_precision, svm_precision, ensemble_precision]
recall_scores = [nb_recall, rf_recall, lr_recall, svm_recall, ensemble_recall]
f1_scores = [nb_f1, rf_f1, lr_f1, svm_f1, ensemble_f1]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12,8))
rects1 = ax.bar(x - 1.5*width, accuracy_scores, width, label='Accuracy')
rects2 = ax.bar(x - 0.5*width, precision_scores, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, recall_scores, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score')

ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

print("THANK YOU")
