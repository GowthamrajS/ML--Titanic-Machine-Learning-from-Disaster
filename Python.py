
####################### Before 2 Process ##

"""
    I am prepared Data in train and test_set Prepared has age_test
    
    
    """ 


# Importing Library

import missingno
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea

z = pd.read_csv("age_test.csv")

q = pd.read_csv("test.csv")

missingno.matrix(q,figsize = [5,6])

# Checking NaN is avalible in Dataset

nullx = z.isnull().sum()

nully = q.isnull().sum()

x = pd.DataFrame()

y = pd.DataFrame()

# finding NaN in Age

ag_nan = z[z.Survived.isin(["NaN"])]


# Removing  Survived Column in Dataset

ag_val = z.dropna(subset = ["Survived"]) 


#   Creating X_train/ Y_train

age_x1 = pd.get_dummies(ag_val.Sex,drop_first=True)

age_x2=  pd.get_dummies(ag_val.Embarked,drop_first =True)

#              3
x = pd.DataFrame()

x["Pclass"]  = ag_val["Pclass"]

x["Age"]  = ag_val["Age"]

x["SibSp"]  = ag_val["SibSp"]

x["Parch"] = ag_val["Parch"]

x["Fare"] = ag_val["Fare"]

con = pd.concat([x,age_x1,age_x2],axis =1)

con.isnull().sum()

x1 = pd.DataFrame()

agl = ag_val.dropna(subset = ["Survived"])

x1["oup"] = agl["Survived"]

x_train = np.matrix(con)

y_train = np.ravel(x1)

# Creating X_test

age_x3 = pd.get_dummies(ag_nan.Sex,drop_first=True)

age_x4=  pd.get_dummies(ag_nan.Embarked,drop_first =True)

y = pd.DataFrame()

y["Pclass"]  = ag_nan["Pclass"]

y["Age"]  = ag_nan["Age"]

y["SibSp"]  = ag_nan["SibSp"]

y["Parch"] = ag_nan["Parch"]

y["Fare"] = ag_nan["Fare"]

cony = pd.concat([y,age_x3,age_x4],axis =1)

cony.isnull().sum()

x_test = np.matrix(cony)


#   Feature Scaling 


from sklearn.preprocessing import StandardScaler

standardcaler =StandardScaler()

x_train= standardscaler.fit_transform(x_train)

x_test = standardscaler.transform(x_test)


#   Model Importing from Sklearn Library

# (criterion ="entropy",n_estimators = 15) 0.94
from sklearn.ensemble import RandomForestClassifier as rf

rf =rf(criterion ="gini",n_estimators = 30)

rf.fit(x_train,y_train)

y_test = rf.predict(x_test)

rf.score(x_train,y_train)

# Exporting ouputput as CSV 

output = pd.DataFrame()

ouput["PassengerId"] = ag_nan["PassengerId"]

ouput["Survived"] = y_test

ouput.to_csv("final_submisssion.csv")

k = q[q.Fare.isin(["NaN"])]


#{'criterion': 'gini', 'n_estimators': 30} # 0.9427







