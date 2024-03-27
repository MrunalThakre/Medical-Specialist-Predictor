import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(5)  # Use any number as the seed

import pickle
# To ignore Warning
import warnings
warnings.filterwarnings('ignore')


# Reading the data
df=pd.read_csv('main.csv')
print(df.head())
y=df['Specialist']
df.drop(columns='Specialist',inplace=True)
X=df[['Age', 'Gender', 'Smoking_Status', 'Alcohol_Consumption', 'Fever',
       'Cough', 'Chest_Pain', 'Headache', 'Shortness_of_Breath', 'Fatigue',
       'Abdominal_Pain', 'Skin_Rash', 'Sore_Throat']]


# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 20% test
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

pickle.dump(model,open('model.pkl','wb'))