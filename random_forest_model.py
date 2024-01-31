##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization

from sklearn.ensemble import RandomForestClassifier #Random Forest Model

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
# Forest Model and hyperparameters
forest_model = RandomForestClassifier(n_estimators = 300, # 100, 500, 1000
                                    max_depth = 17,  # 14, 15, 18
                                    max_features = 6, # 5, 7
                                    max_leaf_nodes = 1800, # 600, 1200
                                    n_jobs= -1,
                                    random_state = 42)
##########################
# Fitting and predicting the data
forest_model.fit(X_train,y_train)
y_pred = forest_model.predict(X_test)
##########################
# Checking the metrics (accuracy and precision )
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='micro'))
##########################
# Perform N-fold cross-validation
n_folds = 5  # Number of folds
scores = cross_val_score(forest_model, X_train, y_train, cv= n_folds, scoring='accuracy')
##########################
# Calculate the mean and standard deviation of the scores
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
##########################
# Calculate predicted probabilities
y_prob = forest_model.predict_proba(X_test)
# Calculate predicted classes(letters)
y_pred = forest_model.predict(X_test)

##########################
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

##########################
# Printing confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel("Predicted ")
plt.ylabel("True")
plt.title("Confusion Matrix Random Forest Model")
plt.show()


