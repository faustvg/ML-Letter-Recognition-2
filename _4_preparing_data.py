##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

##########################
# Importing data
from _2_training_testing import X_train,X_test,y_train,y_test

##########################
# Looking for missing values and duplicated values
print("\nMissing values:\n", X_train.isnull().sum())
##########################
# Drooping attributes that provide no useful information for the task
features_to_drop = ["xy2br", "y-ege", "yegvx", "x2bar","x-bar"]
X_train = X_train.drop(features_to_drop,axis = 1).copy()
X_test = X_test.drop(features_to_drop,axis = 1).copy()
##########################
# Let see again how different features behave now we take out the worst features
# So we again first only for 'A vs not A'.
sns.set(font_scale = 1)
fig, ax = plt.subplots(4, 3, figsize=(15, 15))

features = list(X_train.columns)[1:]

data_plot = X_train.copy()
data_plot['is_A'] = (data_plot['letter'] == 'A')

for name, a in zip(features, ax.ravel()):
  hist = sns.histplot(data=data_plot,
                      x=name,
                      stat='percent',
                      common_norm=False,
                      bins=16,
                      shrink=0.7,
                      discrete=True,
                      hue="is_A",
                      multiple='dodge', 
                      ax = a)
plt.show()
##########################
# Plotting Correlation Heatmap
plt.figure(figsize=(15, 10))
corr_matrix = X_train.iloc[:, 1:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
##########################
# Exporting the data into csv files
X_train = X_train.drop("letter",axis = 1).copy()
X_test = X_test.drop("letter",axis = 1).copy()

X_train.to_csv("final_letter_recognition_X_train.csv",index = False)
X_test.to_csv("final_letter_recognition_X_test.csv",index = False)

y_train.to_csv("final_letter_recognition_y_train.csv",index = False)
y_test.to_csv("final_letter_recognition_y_test.csv",index = False)





