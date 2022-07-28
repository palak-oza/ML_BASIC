import pandas as pd                                         
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.cluster import KMeans
df = pd.read_csv("College_Data")   #readding csv files
df.head()
x=df.iloc[:,:-1].values                         #part of precossing data
y=df.iloc[:,-1].values                          #storing all values of all rows in x and storing the coloumn to be check value in y 
df.info()
from sklearn.model_selection import train_test_split                                        #importing train test split from sklearn to segregate data for training and then testing the program
x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.35,random_state=2)     #assigning the data 
sns.heatmap(df.corr() , cmap=sns.diverging_palette(181, 30, as_cmap=True))
sns.set_style('whitegrid')                   #plotting scatter graph to see if with given codition given clg is private or not
sns.lmplot('Outstate', 'S.F.Ratio', data=df, hue='Private', palette='coolwarm_r', size=6, aspect=1, fit_reg=False)
sns.set_style('whitegrid')                   #plotting scatter graph to see if with given codition given clg is private or not
sns.lmplot('Expend', 'S.F.Ratio', data=df, hue='Private', palette='coolwarm_r', size=6, aspect=1, fit_reg=False)
from sklearn.cluster import KMeans
km = KMeans(2)

df=df.loc[:,~df.columns.str.contains('^Unnamed')]
km.fit(df.drop('Private',axis=1))
df['Private'] =df.Private.astype("category")
df.head()
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(accuracy_score(df.Private, km.labels_))
print(confusion_matrix(df.Private, km.labels_))
print(classification_report(df.Private, km.labels_))
