##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization

from sklearn.neighbors import KNeighborsClassifier # KNeighbors model

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
# KNN model and hyperparameters
knn_model = KNeighborsClassifier(n_neighbors=4,       # 3,5,6
                                algorithm= "kd_tree", # brute
                                weights= "distance",  # uniform
                                n_jobs=-1)

# Fitting and predicting the data
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Checking the metrics (accuracy, precision, and recall)
# Calculate accuracy
print("Accuracy (KNN):", accuracy_score(y_test, y_pred_knn))
print("Precision (KNN):", precision_score(y_test, y_pred_knn, average='micro'))

# Perform N-fold cross-validation
n_folds = 5  # Number of folds
scores_knn = cross_val_score(knn_model, X_train, y_train, cv= n_folds, scoring='accuracy')

# Calculate the mean and standard deviation of the scores
print("Mean accuracy :", scores_knn.mean())
print("Standard deviation :", scores_knn.std())

# Calculate predicted probabilities
y_prob_knn = knn_model.predict_proba(X_test)
# Calculate predicted classes (letters)
y_pred_knn = knn_model.predict(X_test)

# Classification report
print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

# Compute the confusion matrix
conf_mat_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix K-Nearest Neighbors")
plt.show()