# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the necessary packages.  
2. Read the given csv file and display the few contents of the data.  
3. Assign the features for x and y respectively.  
4. Split the x and y sets into train and test sets.  
5. Convert the Alphabetical data to numeric using CountVectorizer.  
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.  
7. Find the accuracy of the model.  

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Nithish Kumar P
RegisterNumber: 212221040113
/*
```
```
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

## Result output
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/12246155-f224-4195-9c0f-52d01289101c)

## data.head()
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/b136f38b-4c8b-4d66-9302-011a5b38d0b4)

## data.info()
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/3f3bc59d-1d10-488a-9710-ea5a8b1331b6)

## data.isnull().sum()
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/be82fc43-2f15-478f-984c-99c5c722c899)

## Y_Prediction value
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/ebb56a0d-0520-4391-b058-fb13f1b90fa4)

## Accuracy value
![image](https://github.com/NaveenKumar-008/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135244/50ac4474-4fd2-4749-b47c-f826df76b608)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
