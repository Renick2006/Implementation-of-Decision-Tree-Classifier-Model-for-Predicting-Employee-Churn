# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## Name: Renick Fabian Rajesh
## Reg No; 212224230227

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Renick Fabian Rajesh
RegisterNumber:  212224230227
*/
```
```
import pandas as pd
data=pd.read_csv('Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("accuracy",accuracy)
print("confusion",confusion)
print("classification",classification)
dt.predict([[10,9,9,66,8,90,90]])
```

## Output:
![WhatsApp Image 2025-11-11 at 00 30 45_1d60c9ac](https://github.com/user-attachments/assets/036093d5-dbd1-4cd2-9421-2acdb5b76ccf)
![WhatsApp Image 2025-12-04 at 18 18 51_6f1b4c54](https://github.com/user-attachments/assets/ad6ecc7a-df60-4d93-bc27-c590ac3d4d75)
![WhatsApp Image 2025-12-04 at 18 19 02_37c6cb77](https://github.com/user-attachments/assets/6b94818d-1b40-473f-8a3d-9c44a14acf8f)
![WhatsApp Image 2025-12-04 at 18 19 16_abaa8b5a](https://github.com/user-attachments/assets/27563dc5-0d7e-4e2d-9883-d6a47f77b32a)
![WhatsApp Image 2025-12-04 at 18 19 32_75ba2c8b](https://github.com/user-attachments/assets/f600d284-9c4e-43d2-85ae-90f1663923f6)
![WhatsApp Image 2025-12-04 at 18 19 47_8b701268](https://github.com/user-attachments/assets/734cb496-8f9a-4a85-9f3b-a25a09160b0e)
![WhatsApp Image 2025-12-04 at 18 20 04_49d56e14](https://github.com/user-attachments/assets/229a8d77-c46d-4fd2-8667-916eaad55f8d)
![WhatsApp Image 2025-12-04 at 18 20 34_76ef1c79](https://github.com/user-attachments/assets/f8fd6589-6471-4c91-84b3-46ade2f3bfca)
![WhatsApp Image 2025-12-04 at 18 20 51_a6d4d311](https://github.com/user-attachments/assets/6f310e26-fa61-4469-8329-610ebdf79bf4)
![WhatsApp Image 2025-12-04 at 18 21 07_48226376](https://github.com/user-attachments/assets/ff76963c-33c5-4bcf-b053-829fd676c923)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
