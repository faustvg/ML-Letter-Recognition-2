#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization

from sklearn.linear_model import LogisticRegression # LogisticRegression model

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

##########################
# Importing data
# Trained data
X_train = pd.read_csv("final_letter_recognition_X_train.csv")
y_train = pd.read_csv("final_letter_recognition_y_train.csv")
# Tested data
X_test = pd.read_csv("final_letter_recognition_X_test.csv")
y_test = pd.read_csv("final_letter_recognition_y_test.csv")
#Reshape into a one-dimensional array, ensuring compatibility with the classification algorithms
y_train = y_train.values.ravel()

##########################
# Logistic Regression model and hyperparameters
lr_model = LogisticRegression(penalty= "l2",  # l1
                            C= 1.0,
                            solver= "saga", # liblinear, sag, newton-cholesky, lbfgs, newton-cg 
                            max_iter= 2000,  # 100,500
                            multi_class= "auto", # auto, ovr(one-vs-rest) and multinomial
                            random_state=42)

# Fitting and predicting the data
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Checking the metrics (accuracy, precision, and recall)
# Calculate accuracy
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision :", precision_score(y_test, y_pred_lr, average='micro'))
# Perform N-fold cross-validation
n_folds = 5
scores_lr = cross_val_score(lr_model, X_train, y_train, cv= n_folds, scoring='accuracy')

# Calculate the mean and standard deviation of the scores
print("Mean accuracy (Logistic Regression):", scores_lr.mean())
print("Standard deviation (Logistic Regression):", scores_lr.std())

# Calculate predicted probabilities
y_prob_lr = lr_model.predict_proba(X_test)
# Calculate predicted classes (letters)
y_pred_lr = lr_model.predict(X_test)

# Classification report
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# Compute the confusion matrix
conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_lr, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Logistic Regression")
plt.show()