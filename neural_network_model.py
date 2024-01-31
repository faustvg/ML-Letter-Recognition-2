##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization

from sklearn.neural_network import MLPClassifier # Neural Network model

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
# Neural Network model and hyperparameters
nn_model = MLPClassifier(hidden_layer_sizes=(100, 100), 
                         max_iter=500, 
                         random_state=42)

# Fitting and predicting the data
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# Checking the metrics (accuracy, precision, and recall)
# Calculate accuracy
print("Accuracy (Neural Network):", accuracy_score(y_test, y_pred_nn))
print("Precision (Neural Network):", precision_score(y_test, y_pred_nn, average='micro'))

# Perform N-fold cross-validation
n_folds = 5  # Number of folds
scores_nn = cross_val_score(nn_model, X_train, y_train, cv=n_folds, scoring='accuracy')

# Calculate the mean and standard deviation of the scores
print("Mean accuracy (Neural Network):", scores_nn.mean())
print("Standard deviation (Neural Network):", scores_nn.std())

# Calculate predicted probabilities
y_prob_nn = nn_model.predict_proba(X_test)
# Calculate predicted classes (letters)
y_pred_nn = nn_model.predict(X_test)

# Classification report
print("Classification Report (Neural Network):")
print(classification_report(y_test, y_pred_nn))

# Compute the confusion matrix
conf_mat_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_nn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Neural Network")
plt.show()

