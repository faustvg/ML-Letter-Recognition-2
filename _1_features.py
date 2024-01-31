##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################
#Importing the first libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creation of univariate and bivariate plots
import matplotlib.pyplot as plt # comprehensive data visualization
##########################
# Column names
column_names = [
    'letter', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar',
    'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'
]
#############################################################################
# Convert the data to a format you can easily manipulate (without changing the data itself), e.g. a Pandas DataFrame
data = pd.read_csv('letter-recognition.data', delimiter=',', names=column_names)
data.to_csv("letter-recognition.csv",index= False)
######################
# Looking at the distribution of the different features from the data
print(data.info()) # Basic Information about the data
print("\nSample (10)\n",data.head(10)) # View a sample data (100)
print("\nStatistics information\n",data.describe()) # Statistic information about the data
##########################
#Sorted data
sorted_data = data.sort_values('letter')
##########################
# Count of each letter(sorted)
print("\n",sorted_data['letter'].value_counts().sort_index())
##########################
# Visualization
sns.countplot( x = sorted_data['letter'])
plt.xlabel('Letter')
plt.ylabel('Count')
plt.title('Count of Letters Sorted')
plt.show()
data.hist(bins=40,figsize=(12, 8)) #  
plt.show() 