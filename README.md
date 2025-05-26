# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RESHMA C
RegisterNumber:  212223040168
*/
```
```
from google.colab import files
uploaded = files.upload()

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/5ba32273-9b4f-4e56-b17e-6eb71c3db8e4)

![image](https://github.com/user-attachments/assets/bb4732a7-d9f7-4df3-a930-a4370eba0c80)

![image](https://github.com/user-attachments/assets/371be2de-5f44-4aa6-bb58-5381870db822)

![image](https://github.com/user-attachments/assets/d945fda8-5f7e-45cc-81cb-1ab50b8ced76)

![image](https://github.com/user-attachments/assets/e48c56ba-7e33-4aa5-b000-91dd6eb736a8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
