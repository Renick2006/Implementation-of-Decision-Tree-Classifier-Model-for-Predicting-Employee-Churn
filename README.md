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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("heart (1).csv")
print("Dataset Preview:")
df.head()
print("Missing Values:")
df.isnull().sum()
X=df.drop("target",axis=1)
y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
dt_model=DecisionTreeClassifier(random_state=42)
dt_model=dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
accuracy_dt=accuracy_score(y_test,y_pred_dt)
print("\nDecision Tree Accuracy:",accuracy_dt)
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
accuracy_rf=accuracy_score(y_test,y_pred_rf)
print("Random Forest Accuracy :",accuracy_rf)
print("\nDecision Tree Classification Report : \n",classification_report(y_test,y_pred_dt))
print("\nRandom Forest Classification Report : \n",classification_report(y_test,y_pred_rf))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test,y_pred_dt),annot=True,fmt='d',cmap='Blues')
plt.title('Decision Tree Confusion Matrix')

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test,y_pred_rf),annot=True,fmt='d',cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()
if accuracy_rf>accuracy_dt:
    print("Random Forest performs better than Decision Tree")
else:
    print("Decision Tree performs better than Random Forest")
```

## Output:
<img width="831" height="293" alt="Screenshot 2025-09-25 161641" src="https://github.com/user-attachments/assets/ea41d464-e89c-498f-a1e1-83361b303672" />
<img width="184" height="410" alt="Screenshot 2025-09-25 161647" src="https://github.com/user-attachments/assets/5a1fd331-750d-486e-8236-fd7ee66838a8" />
<img width="466" height="41" alt="Screenshot 2025-09-25 161654" src="https://github.com/user-attachments/assets/a95e4e65-2ea8-4656-88a7-322f9e1f4b0a" />
<img width="489" height="44" alt="Screenshot 2025-09-25 161706" src="https://github.com/user-attachments/assets/d0b1f36d-f7f7-4039-ab76-6d5dc0550de3" />
<img width="624" height="509" alt="Screenshot 2025-09-25 161714" src="https://github.com/user-attachments/assets/176fbe03-f744-4c6a-9b35-d58c2a9d9624" />
<img width="1440" height="647" alt="Screenshot 2025-09-25 161727" src="https://github.com/user-attachments/assets/4f948913-6022-4e85-bc1f-d214a3ad1287" />
<img width="533" height="25" alt="Screenshot 2025-09-25 161734" src="https://github.com/user-attachments/assets/29a0c6de-6b8c-4a1e-9e13-1e43d40773d2" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
