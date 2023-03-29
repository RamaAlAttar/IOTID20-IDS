#© 2022 Rama Attar <ram20208027@std.psut.edu.jo>#

from datetime import datetime
import os
import numpy as np
from numpy import mean,std
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_validate,  StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
import time
from sklearn.metrics import classification_report, accuracy_score, make_scorer, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.feature_selection import mutual_info_classif


def classification_report_with_accuracy_score(y_true, y_pred):

    print (classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

################################################################################################DATASET IMPORT##########################################################################################################

Dataset_File = 'IoT Network Intrusion Dataset.csv'
dataset = pd.read_csv(Dataset_File)

##############################################################################################DATASET PREPROCESSING######################################################################################################

print(dataset.dtypes)

### Infinity Values Clean Up ###

IsInf= dataset.isin([np.inf, -np.inf]).sum()

#print(IsInf.sort_values(ascending=False).head())

###Replacing Infinity Values

dataset.replace([np.inf, -np.inf], np.nan,inplace=True)

### Missing values clean up using the mean startegy###

IsNa = dataset.isna().sum()
IsNull = dataset.isnull().sum()
#print(IsNa.sort_values(ascending=False).head())
#print(IsNull.sort_values(ascending=False).head())

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(dataset[['Flow_Pkts/s','Flow_Byts/s']])
dataset[['Flow_Pkts/s','Flow_Byts/s']] = imputer.transform(dataset[['Flow_Pkts/s','Flow_Byts/s']])

for col_name in dataset.columns:
    if dataset[col_name].dtypes == 'object' :
        unique_cat = len(dataset[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

######### CHANGING CATEGORICAL VALUES INTO BINARY############

# Label
categorical_column_Label=['Label']
dataset_categorical_values_Label = dataset[categorical_column_Label]
unique_Label=sorted(dataset.Label.unique())
string1 = 'Label_'
unique_Label2=[string1 + x for x in unique_Label]
# Cat
categorical_column_Cat=['Cat']
dataset_categorical_values_Cat = dataset[categorical_column_Cat]
unique_Cat=sorted(dataset.Cat.unique())
string2 = 'Cat_'
unique_Cat2=[string2 + x for x in unique_Cat] 
# Sub Cat
categorical_column_Sub_Cat=['Sub_Cat']
dataset_categorical_values_Sub_Cat = dataset[categorical_column_Sub_Cat]
unique_Sub_Cat=sorted(dataset.Sub_Cat.unique())
string3 = 'Sub_Cat_'
unique_Sub_Cat2=[string3 + x for x in unique_Sub_Cat]



cols = ['Label', 'Cat', 'Sub_Cat']
#
# Encode labels of multiple columns at once
#
dataset[cols] = dataset[cols].apply(LabelEncoder().fit_transform)
#
# Print head
#
print(dataset.head())
print(dataset.dtypes)
print(dataset.dtypes)

dataset.drop('Flow_ID', axis=1, inplace=True)
dataset.drop('Dst_IP', axis=1, inplace=True)
dataset.drop('Src_IP', axis=1, inplace=True)
dataset.drop('Timestamp', axis=1, inplace=True)

##############################################################################################MACHINE LEARNING-BINARY######################################################################################################
dataset.drop('Cat', axis=1, inplace=True)
dataset.drop('Sub_Cat', axis=1, inplace=True)

X = dataset.iloc[:, 0:78]
y = dataset.iloc[:, 79]
dtc =  DecisionTreeClassifier ()
rfc = RandomForestClassifier()
et_clf = ExtraTreesClassifier()
knn=KNeighborsClassifier()

clf = [('knn',knn),('et_clf',et_clf),('rfc',rfc),('dtc',dtc) ]


kf = StratifiedKFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index] 

    model = StackingClassifier( estimators = clf,final_estimator = rfc)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    print(f'For fold {fold}:')
    print("time before test", datetime.now())
    print(f'Accuracy Test: {model.score(X_test, y_test)}')
    print("time after test", datetime.now())
    print(f'Accuracy Train: {model.score(X_train, y_train)}')
    print('', classification_report_with_accuracy_score(y_test, y_pred))

    ##############################################################################################MACHINE LEARNING-CATEGORY######################################################################################################
dataset.drop('Label', axis=1, inplace=True)
dataset.drop('Sub_Cat', axis=1, inplace=True)

X = dataset.iloc[:, 0:78]
y = dataset.iloc[:, 79]
dtc =  DecisionTreeClassifier ()
rfc = RandomForestClassifier()
et_clf = ExtraTreesClassifier()
knn=KNeighborsClassifier()

clf = [('knn',knn),('et_clf',et_clf),('rfc',rfc),('dtc',dtc) ]


kf = StratifiedKFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index] 

    model = StackingClassifier( estimators = clf,final_estimator = rfc)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    print(f'For fold {fold}:')
    print("time before test", datetime.now())
    print(f'Accuracy Test: {model.score(X_test, y_test)}')
    print("time after test", datetime.now())
    print(f'Accuracy Train: {model.score(X_train, y_train)}')
    print('', classification_report_with_accuracy_score(y_test, y_pred))
##############################################################################################MACHINE LEARNING-SUBCATEGORY######################################################################################################
dataset.drop('Label', axis=1, inplace=True)
dataset.drop('Cat', axis=1, inplace=True)

X = dataset.iloc[:, 0:78]
y = dataset.iloc[:, 79]
dtc =  DecisionTreeClassifier ()
rfc = RandomForestClassifier()
et_clf = ExtraTreesClassifier()
knn=KNeighborsClassifier()

clf = [('knn',knn),('et_clf',et_clf),('rfc',rfc),('dtc',dtc) ]


kf = StratifiedKFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index] 

    model = StackingClassifier( estimators = clf,final_estimator = rfc)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    print(f'For fold {fold}:')
    print("time before test", datetime.now())
    print(f'Accuracy Test: {model.score(X_test, y_test)}')
    print("time after test", datetime.now())
    print(f'Accuracy Train: {model.score(X_train, y_train)}')
    print('', classification_report_with_accuracy_score(y_test, y_pred))
