# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:54:51 2020

@author: masud
"""

import matplotlib.pyplot as plt
import seaborn as sns



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Importing the splitter, classification model, and the metric
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Importing SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler, RobustScaler

#import seaborns as sns



# Import the data
data = pd.read_csv('F:\BUET\Data Mining\Assignment 1\Dataset\creditcard.csv')
data.head()

target_count = data['Class'].value_counts()
#print(target_count)

# Initially Class ratio
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 0), ': 1')

#target_count.plot(kind='bar', title='Count (target)');


# Here we use Robust Scaler technique for feature scalling
# Scale "Time" and "Amount"



data['scaled_amount'] = RobustScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = RobustScaler().fit_transform(data['Time'].values.reshape(-1,1))


#Time Amount
df_example = data[['V1', 'V2', 'Class']]
#sns.scatterplot(data = data, x ='V1', y = 'V2', hue = 'Class')

#plt.scatter(data = data, x ='CreditScore', y = 'Age',label='Exited')
#plt.show()


#Splitting the data with stratification
X_train, X_test, y_train, y_test = train_test_split(df_example[['V1', 'V2']], data['Class'], test_size = 0.3, stratify = data['Class'], random_state = 101)


#Create an oversampled training data
smote = SMOTE(random_state = 101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

#Creating a new Oversampling Data Frame
df_oversampler_X = pd.DataFrame(X_oversample, columns = ['V1', 'V2'])
#data['Exited']
#df_oversampler['Exited']
df_oversampler_y = pd.DataFrame(y_oversample)
df=pd.concat([df_oversampler_X,df_oversampler_y],axis=1)


## After SMOTE Technique test data ratio
af_smot= y_oversample.value_counts()
print('Class 0:', af_smot[0])
print('Class 1:', af_smot[1])
print('Proportion:', round(af_smot[0] / af_smot[1], 2), ': 1')

#af_smot.plot(kind='bar', title='Count (target)');



df_example = data[['V1', 'V2', 'Class']]
sns.scatterplot(data = df, x ='V1', y = 'V2', hue = 'Class')

#Training with imbalance data
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print(classification_report(y_test, classifier.predict(X_test)))


