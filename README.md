
# Loan Prediction Model

📚 This code represents a loan prediction model using Support Vector Machine (SVM) algorithm. The purpose of this model is to predict the approval status of a loan application based on various features.

## Prerequisites

Make sure you have the following libraries installed before running the code:
- 📦 numpy 
- 📦 pandas 
- 📦 seaborn 
- 📦 scikit-learn 

## Dataset

The code assumes that the dataset is available in CSV format. The dataset used for training and testing the model is loaded into a pandas DataFrame using the `read_csv()` function.

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the dataset to pandas DataFrame
df = pd.read_csv('/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
df.head()
df.describe()
```

## Data Preprocessing

The dataset is preprocessed to handle missing values and convert categorical columns into numerical values. The following steps are performed:

```python
# number of missing values in each column
df.isnull().sum()

# dropping the missing values
df = df.dropna()
df.isnull().sum()
df.head()
```

Dependent column values
```python
df['Dependents'].value_counts()
```

Replacing the value of 3+ to 4
```python
df = df.replace(to_replace='3+', value=4)
df['Dependents'].value_counts()
```

Education & Loan Status
```python
sns.countplot(x='Education',hue='Loan_Status',data=df)
```

Marital status & Loan Status
```python
sns.countplot(x='Married',hue='Loan_Status',data)
df.head(10)
```

Convert categorical columns to numerical values
```python
df.replace({"Gender":{'Male':1,'Female':0},
           "Education":{'Graduate':1,'Not Graduate':0},
           "Property_Area":{'Rural':0,'Semiurban':1,'Urban':2},
           "Self_Employed":{'Yes':1,'No':0},
           "Married":{'Yes':1,'No':0}},inplace=True)
```

```python
df.head(10)
```

## Model Training and Evaluation

Separating the data and label
```python
x = df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df['Loan_Status']

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=42)
classifier= svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
```

Accuracy score on training data
```python
x_train_prediction = classifier.predict(x_train)
train_data_acc = accuracy_score(x_train_prediction,y_train)
print(f'Accuracy on training data : {train_data_acc}')
```

Accuracy score on testing data
```python
x_test_prediction = classifier.predict(x_test)
test_data_accuray = accuracy_score(x_test_prediction,y_test)
print(f'Accuracy on testing data : {test_data_accuray}')
```

🚨 Note: This code is provided as a prototype and may require further enhancements and optimizations for real-world scenarios. 

📝 Copy the code directly to your GitHub repository and customize it according to your specific requirements.
