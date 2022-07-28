import pandas as pd                                           #importing pandas,matplotlib,numpy,seaborn,sklearn library by renaming then in short for easy usage
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics                                    #for easy accecebility of matrix
%matplotlib inline                                             #for plotting the graph in same cell with code instead of a new one
df=pd.read_csv('heart.csv')                                 #reading heart csv file and stoing it in df
df.head()  
print(f"rows and coloumn:{df.shape}")           #printing the number of coloumn and rows in given file
x=df.iloc[:,:-1].values                         #part of precossing data
y=df.iloc[:,-1].values                          #storing all values of all rows in x and storing the coloumn to be check value in y 
df.info()                                       #getting brief information abt the file(attributes)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder  #preprocessing the data and importing files needed
labelencoder=LabelEncoder()                                   #intialising labelencoder with the module 
#x[:,2]=labelencoder.fit_transform(x[:,2])
#x[:,12]=labelencoder.fit_transform(x[0,12])                  #using labelencoder to assign values of string as int for easy comparison
#x[:,13]=labelencoder.fit_transform(x[0,13])
onehotencoder=OneHotEncoder()                                 #converts all the string data in int for easy comparison
x=onehotencoder.fit_transform(x).toarray()
print(x)
sns.heatmap(df.corr() , cmap=sns.diverging_palette(175, 20, as_cmap=True)) #formation of graph for easy visual comaprison of data
from sklearn.model_selection import train_test_split                                 #importing train test split from sklearn to segregate data for training and then testing the program
x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.35,random_state=2) #assigning the data
print(x_train.shape)                    #showing values of rows and column after splitting of data
from sklearn.linear_model import LogisticRegression        #importing logistic regression from linear model in sklearn
reg=LogisticRegression()                                   #assigning the logistic regression to reg
reg.fit(x_train,y_train)                                   #fitting data 
print(reg.predict(x_test[0].reshape(1,-1)))                #to check one output of model 
pred=reg.predict(x_test)                                  #applying predict to all rows or data
score=reg.score(x_test,y_test)
print("The accuracy of the model is :- ",(score*100))     #calculation of accuracy of the model
from sklearn.metrics import confusion_matrix              #improting confusion matrix from sk learn to segrate the comparision of actual and predict data
cm=confusion_matrix(y_test,pred)
print(cm)
plt.figure(figsize=(10,5))                                                            #plotting map of confusion matrix for user to have easy visualisation
sns.heatmap(cm , annot=True , fmt='.3f' , linewidths=1 , square=True , cmap='Blues_r')
plt.ylabel("TRUE HEART DISEASE")
plt.xlabel("PREDICTED HEART DISEASE")
sampletitle=f"ACCURACY OF BUILT MODEL: {score}"
plt.title(sampletitle,size=15)
