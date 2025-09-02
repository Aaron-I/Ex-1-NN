<H3>NAME: Aaron I</H3>
<H3>REGISTER NUMBER: 212223230002</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 25.08.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM & OUTPUT:

```
#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

```
#Read the dataset from drive
df = pd.read_csv('Churn_Modelling.csv')
print(df)
```

<img width="1464" height="664" alt="Screenshot 2025-09-02 082514" src="https://github.com/user-attachments/assets/ef144bee-f8bb-4646-81f8-c6993ec5fb6c" />


```
#Split the dataset
X = df.iloc[:,:-1].values
print(X)
Y = df.iloc[:,-1].values
print(Y)
```

<img width="729" height="205" alt="image" src="https://github.com/user-attachments/assets/62eebadf-dda3-4b3e-ad4a-d00f71d0dcb3" />


```
# Finding Missing Values
print(df.isnull().sum())
```

<img width="791" height="401" alt="image" src="https://github.com/user-attachments/assets/dea722f0-dff6-432b-8420-25b0fc41e81e" />

```
#Dropping string values data from dataset
df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()
```

<img width="1322" height="329" alt="image" src="https://github.com/user-attachments/assets/8b9bf4b3-e006-4d8a-aee1-4c43d5127dc9" />

```
#Handling Missing values
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
Y = df.iloc[:,-1].values
print(Y)
```

<img width="824" height="441" alt="image" src="https://github.com/user-attachments/assets/4e7ca8d6-e575-441e-ad60-1f010410d420" />

```
#Check for Duplicates
df.duplicated()
```

<img width="806" height="525" alt="image" src="https://github.com/user-attachments/assets/c0c1720a-a56f-4cc9-8268-0661c935e6a0" />

```
#Detect Outliers
print(df.describe())
```

<img width="984" height="646" alt="image" src="https://github.com/user-attachments/assets/a3b3ccac-a235-4c98-94d0-ee91e149200e" />

```
#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
```

<img width="956" height="609" alt="image" src="https://github.com/user-attachments/assets/894eb48c-4113-4948-b4d8-465f083f0064" />

```
#split the dataset into input and output
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```

<img width="859" height="439" alt="image" src="https://github.com/user-attachments/assets/d51a6d15-6c86-496f-896e-3adcecb029d4" />

```
#splitting the data for training & Testing
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
```

```
#Print the training data and testing data
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

<img width="972" height="660" alt="image" src="https://github.com/user-attachments/assets/cb7ebd44-b190-46fd-aca1-30e6fc568882" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


