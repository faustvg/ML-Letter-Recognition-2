##########################
#Letter Image Recognition Data
# Faustino Vazquez Gabino| 203961
# Alessandro Sapia       | 290742
##########################
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

data = pd.read_csv('letter-recognition.csv') # Importing the DataFrame
y = data['letter'] # Taking the letter column as our target value 

# Training the first 16,000 entries and testing the rest 4000 as it was suggested
X_train, X_test = train_test_split(data, test_size= 0.2 , random_state = 42) 
y_train, y_test = train_test_split(y, test_size= 0.2, random_state= 42 )
