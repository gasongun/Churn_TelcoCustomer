"""
Churn Modelling

This project aim is find the persons who will leave the company by using the churn dataset that located on belowed link.
Telco churn dataset gives third-quarter Demographic and Service information for 7043 customers of a California telecom company. It gives which customers left the company after using some services.


Original file is located at
    https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113


We are going to follow the content below in this project:

1- Import Data and Libraries
    a- Importing Libraries
    b- Importing Data
2- Data Preprocessing
3- Feature Engineering
    a- Missing Value Analysis
    b- Outlier Value Analysis
    c- Feature Extraction
    d- Label Encoding
    e- One Hot Encoding
4- Find the optimum model for estimate possible churn customers
    a-  Hyperparameter optimization

"""
## Import Data and Libraries

# Importing Libraries

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Importing Data

def load_dataset(data,pathChange = True):
    path = os.getcwd()
    if pathChange == False:
        pathContinue = input("Write the continue of the path")
        os.chdir(path + '/' + pathContinue)
        path = os.getcwd()
    return pd.read_csv(path + '/' + data + ".csv")


df_ = load_dataset("Telco-Customer-Churn",False)     # DSMBLC8\Modul6_MakineOgrenmesi
df = df_.copy()
df.head()


## Data Preprocessing


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    This function returns the group of variables in the dataset. These are categorical, numerical and cardinal which is defined as object but number of categorical values more than the threshold value.
    PS: Categorical group includes that looks numeric but is categorical variables.


    Parameters
    ------
        dataframe: dataframe
                Dataframe which uses for variable types
        cat_th: int, optional
                Threshold value for a variable that looks numeric but is categorical.
        car_th: int, optional
                Threshold value for a variable that looks categoric but is cardinal.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of categorical but cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = All variables in the dataset.
        cat_cols list includes num_but_cat list

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# We have 21 variables and 7043 observations

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# When we examine the variable groups, there is a problem on cat_but_car group. TotalCharges variable has to be with the numerical values then first we have to change this problem.

print(f'cat_cols: {cat_cols}\n')
# cat_cols: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn', 'SeniorCitizen']
print(f'num_cols: {num_cols}\n')
# num_cols: ['tenure', 'MonthlyCharges']
print(f'cat_but_car: {cat_but_car}\n')
# cat_but_car: ['customerID', 'TotalCharges']


"""  TotalCharges will change to numeric but in pd.to_numeric function, there is parameter about the errors. 
Default is "raise", if every item is same type in variable, it will change to numeric type. However in our situation, it errors which is "Unable to parse string" because of the white spaces.
If we choose "ignore", these white spaces will remain and variable will be object type again.
In these situation we choose "coerce", therefore white spaces will change to NaN. 

errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
        - If 'ignore', then invalid parsing will return the input.
"""

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# After TotalCharges changes, we execute the grab_col_names function again and check is it comes correctly.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(f'num_cols: {num_cols}\n')
# num_cols: ['tenure', 'MonthlyCharges', 'TotalCharges']

# Churn is a dependent variable which values take Yes/No. We change to 1/0 to use in data understanding stage.

df["Churn"].value_counts()
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

# Categorical Variable Analysis

def cat_summary(dataframe, col_name, plot=False):
    """
    This function returns value counts and ratios of categorical variables in dataset. Also when you choose plot parameter as True, it gives count plot for each categorical variable.

    Parameters
    ----------
    dataframe: Dataframe which uses for analyzing the categorical variables
    col_name: Column which to be analyzed
    plot: Parameter takes True/False option that if its true, it gives a countplot of the variable

   Examples
   ------
       import seaborn as sns
        df = sns.load_dataset("diamonds")
        cat_summary(df, "color", plot=True)

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# We examine our categorical variables with the cat_summary function.
# After examination, we keep in mind %90 percent of PhoneService variable is Yes and only %27 percent of Churn variable which is a dependent variable is churn.

for col in cat_cols:
    cat_summary(df, col, plot=True)

"""
     PhoneService  Ratio
Yes          6361 90.317
No            682  9.683
##########################################
   Churn  Ratio
0   5174 73.463
1   1869 26.537
"""

def num_summary(dataframe, numerical_col, plot=False):
    """
    This function returns description of numerical variable in dataset.
    Quartiles are split more frequently to examine the outlier and extreme values.
    Also when you choose plot parameter as True, it gives histrogram plot for numerical variable.

    Parameters
    ----------
    dataframe: Dataframe which uses for analyzing the numerical variables
    numerical_col: Column which to be analyzed
    plot: Parameter takes True/False option that if its true, it gives a histogram of the variable

    Examples
     ------
       import seaborn as sns
        df = sns.load_dataset("diamonds")
        num_summary(df, "carat", plot=True)

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

# We examine our numercical variables with the num_summary function.
# When we look at the Tenure variable on graphics, we see the most frequent months are 1 and 70 months. We have three different type of contract in a dataset, that may be cause of these contract type differences.

for col in num_cols:
    num_summary(df, col, plot=True)


# Examination of Contract types with graphics.

# When we look at the distribution of the tenure with a contract types, there is an obvious difference between "Month-to-month" and yearly contracts.
"""1 aylık ve iki yıl üzeri kontratlardan kaynaklı tenure içerisinde bir dağılım farklılığı varmış. Bunun aslında modelde etkili olmasını bekleriz."""

df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
df[df["Contract"] == "One year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("One year")
plt.show()


## Feature Engineering

# Missing Value Analysis
# Looks like only the TotalCharges variable has missing values.

df.isnull().sum().sort_values(ascending=False)

# But! When we look at the index of those whose total charges are empty in the tenure variable, we see that the zero ones in tenure are also empty.

df["tenure"][df["TotalCharges"].isnull()]
df["tenure"][df["tenure"] == 0] = None
df.isnull().sum().sort_values(ascending=False)

# For missing value imputation, we look at the distribution of the variable. TotalCharges variable is a skewed, then we choose median imputation for the missing values.
# We use the same imputation technic's for tenure but we could choose drop these cases.
df["TotalCharges"].hist(bins=20)
plt.xlabel("TotalCharges")
plt.show()

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True) # 1397.475
df["tenure"].fillna(df["tenure"].median(), inplace=True) # 29.0

# Outlier Value Analysis

# If there are outliers in dataset, these are not generalizable and confuse the model.
# Before start to building the model, we have to control their existance and if they are exist, we have to take some actions about them.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# There is no outlier value.

for col in num_cols:
    print(col, check_outlier(df, col))

# Feature Extraction

# *CustomerTime* variable from Tenure
# First examine the relationship between tenure and churn variables.
# Churn ratio in the first 12 month is more than the longer months. This shows us, a customer churn probability is getting less with more tenure time.
# We create a new variable as a CustomerTime with these consequences.

bins = 50
plt.hist(df[df['Churn'] == 1].tenure, bins, alpha=0.5, density=True, label='Churned')
plt.hist(df[df['Churn'] == 0].tenure, bins, alpha=0.5, density=True, label="Didn't Churn")
plt.legend(loc='upper right')
plt.show()

df.loc[(df['tenure'] <= 12), 'CustomerTime'] = '1year'
df.loc[(df['tenure'] > 12) & (df['tenure'] <= 48), 'CustomerTime'] = '4years'
df.loc[(df['tenure'] > 48), 'CustomerTime'] = '4years+'

df['CustomerTime'].value_counts()
"""
4years     2629
4years+    2239
1year      2175
Name: CustomerTime, dtype: int64"""

# *PaymentMethod_New* variable from PaymentMethod
# There are 4 group in the PaymentMethod variable but two of them relative with bank, other two of them relative with check.
# We combine these two groups together and create a new variable as a PaymentMethod_New.

df.loc[(df['PaymentMethod'] == 'Bank transfer (automatic)') | (df['PaymentMethod'] == 'Credit card (automatic)'), 'PaymentMethod_New'] = 'Bank'
df.loc[(df['PaymentMethod'] == 'Mailed check') | (df['PaymentMethod'] == 'Electronic check'), 'PaymentMethod_New'] = 'Check'

df['PaymentMethod_New'].value_counts()
"""
Check    3977
Bank     3066
Name: PaymentMethod_New, dtype: int64"""

# *LongTermContract* variable from Contract
# Customers are who have a long-term contract looks like has less churn probability. With this inference "One year" and "Two year" contracts are more similar.
# We create a new variable as a LongTermContract from this inference.

df["LongTermContract"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)
df['LongTermContract'].value_counts()
"""
0    3875
1    3168
Name: LongTermContract, dtype: int64"""

# *MaxPackageInternet* variable from InternetService and all related variables.
# InternetService variable has two type that Fiber Optic and DSL.
# There are 6 variables in the dataset that are related to the InternetService variable which these are "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
# If customer has all of them, that means the customer uses all of our services.
# We create a new variable as a MaxPackageInternet, based on the cases where all these services are Yes.

df['InternetService'].value_counts()

df.loc[:, "MaxPackageInternet"] = np.where((df["InternetService"] != 'No') & (df["OnlineSecurity"] == 'Yes')
                                               & (df["OnlineBackup"] == 'Yes') & (df["DeviceProtection"] == 'Yes')
                                               & (df["TechSupport"] == 'Yes') & (df["StreamingTV"] == 'Yes') & (df["StreamingMovies"] == 'Yes'), '1','0')

df['MaxPackageInternet'].value_counts()
"""
0    6759
1     284
Name: MaxPackageInternet, dtype: int64 """

# *noSup* variable from all support variables.
# There are some support services variables in the dataset which are "OnlineBackup", "DeviceProtection", "TechSupport".
# If customer takes one of them the support services, they have to be less churn probability.
# We create a new variable as a noSup, based on the cases where one of these services are Yes.

df["noSup"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)
df['noSup'].value_counts()
"""
1    6317
0     726
Name: noSup, dtype: int64"""

# *TotalServices* variable from all of the service variables.
# We create a new variable as a TotalServices, based on the number of services our customers have.

df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

df['TotalServices'].value_counts()
"""
1    2253
4    1062
3    1041
2     996
5     827
6     525
7     259
0      80
Name: TotalServices, dtype: int64"""

# *AvgPerMonth* variable from TotalCharges and tenure variables.
# TotalCharges variable show us what is the total income for each customer and tenure shows us how many months have been in a customer relationship with the company.
# In this context, when we take proportion of both variables, it means customer's monthly average bill.

df["AvgPerMonth"] = df["TotalCharges"] / df["tenure"]

# *CurrentIncrease* variable from AvgPerMonth
# AvgPerMonth variable gives us the customer's average bill for month.
# If we calculate AvgPerMonth and MonthlyCharges proportion, this allows us to understand that the bill has reduced or raised at what rate.

df["CurrentIncrease"] = df["AvgPerMonth"] / df["MonthlyCharges"]

# *StreamingService* variable from StreamingTV and StreamingMovies variables.
# There are two variable about streaming system in the dataset.
# We create a new variable as a StreamingService, based on the cases where all streaming services are Yes.

df["StreamingService"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

df['StreamingService'].value_counts()


# There are 2 variables that we produce using variables themselves. We have to remove these 2 variables.
# At the begining of the work, there were 21 variables, now its 28.
dropVariables = ["PaymentMethod","Contract"]
df.drop(dropVariables,axis=1,inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Label Encoding

# This library helps to change the labels of the variables between 0 to number of classes-1.
# We use one hot encoding for variables have more than two groups.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# One Hot Encoding

# This library helps to create a variable for each variable groups.
# We already changed the binary variables, now it's time for variables wtih has more than 2 groups.
# cat_cols include Churn and TotalServices, we exclude these two variables. Because one is our dependent variable and other one is count of services which is a continuous variable.

cat_cols
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "TotalServices"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

# Little control for missing values
df.isnull().sum().sort_values(ascending=False)

## Modelling

# LightGBM has the most accuracy and F1 score. F1 score is important for the imbalanced data. Because accuracy shows correctly classified groups but in the imbalanced dataset, this misleads us.

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

models = [('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=42)),
          ('RF', RandomForestClassifier(random_state=42)),
          ('SVM', SVC(gamma='auto', random_state=42)),
          ('XGB', XGBClassifier(random_state=42)),
          ("LightGBM", LGBMClassifier(random_state=42))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 3)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 3)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 3)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 3)}")

"""
########## KNN ##########
Accuracy: 0.764
Recall: 0.454
Precision: 0.571
F1: 0.505
########## CART ##########
Accuracy: 0.731
Recall: 0.512
Precision: 0.493
F1: 0.502
########## RF ##########
Accuracy: 0.792
Recall: 0.494
Precision: 0.641
F1: 0.558
########## SVM ##########
Accuracy: 0.768
Recall: 0.274
Precision: 0.653
F1: 0.386
########## XGB ##########
Accuracy: 0.786
Recall: 0.494
Precision: 0.622
F1: 0.551
########## LightGBM ##########
Accuracy: 0.795
Recall: 0.524
Precision: 0.639
F1: 0.575
"""

# Hyperparameter optimization

lgbm_model = LGBMClassifier(random_state=13)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=13).fit(X, y)
cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.8034
cv_results['test_f1'].mean()       # 0.5720
cv_results['test_roc_auc'].mean()  # 0.8458

# We control our model again after hyperparameter optimization, because of the imbalance our model unsuccessful to predict churn customer.
# That means we cannot detect the customer who will leave the company.
y_pred = lgbm_final.predict(X)
print(confusion_matrix(y, y_pred))
"""
[[4828  346]
 [ 816 1053]]
 """

# We have to balance the data for more accurate predicts on the next stage of the project. But i'm not doing now :)