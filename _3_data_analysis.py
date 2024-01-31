##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization
##########################
# Importing data
from _2_training_testing import X_train
trained_data = X_train
##########################
# Plotting Histogram
trained_data.hist(figsize = (12,8))

# Correlation Plot
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Plot')
plt.show()

# Bar Chart for Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='letter', data=data)
plt.title('Class Distribution')
plt.show()


##########################
# Now, let us see how the different features behave for the different letters
# This is not so easy to do in an itretpretable way for all 26 letters
# at the same time, so we do it here only for 'A vs not A'.
sns.set(font_scale = 1)
fig, ax = plt.subplots(4, 4, figsize=(15, 15))

features = list(trained_data.columns)[1:]

data_plot = trained_data.copy()
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
corr_matrix = trained_data.iloc[:, 1:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


