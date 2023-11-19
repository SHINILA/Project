# Project
#!/usr/bin/env python
# coding: utf-8

# # Marketing Campaign

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('marketing_trn_data.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df1 = pd.read_csv('marketing_trn_class_labels.csv')


# In[6]:


print('Number of rows: ' + str(df.shape[0]) + '. Number of columns: ' + str(df.shape[1]))


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


df.dtypes


# In[11]:


df.dtypes.value_counts()


# #  Cleaning Preprocessing

# In[12]:


import warnings

warnings.filterwarnings('ignore')


# In[13]:


df = df.drop('Dt_Customer', axis=1)


# In[14]:


df[' Income ']=df[" Income "].str.replace('[\$\,]',"")


# In[15]:


df[' Income '] = df[' Income '].astype(float)


# In[16]:


df[' Income '] = pd.to_numeric(df[' Income '], errors='coerce').astype('Int64')


# In[17]:


df.dtypes


# In[18]:


df[' Income '].sum()


# In[19]:


for column in df.columns:
    print(column.capitalize() + ' - Missing Values: ' + str(sum(df[column].isnull())))


# In[20]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Handling Missing value

# In[22]:


plt.figure(figsize=(12, 7))
sns.boxplot(x= ' Income ',y='Education',data=df,palette='winter')


# In[23]:


def impute_income(cols):
    Income = cols[0]
    Education = cols[1]
    
    if pd.isnull(Income):

        if Education == 1:
            return 37

        elif Education == 2:
            return 29

        else:
            return 24

    else:
        return Income  

df[' Income '] = df[[' Income ','Education']].apply(impute_income,axis=1)


# In[24]:


for column in df.columns:
    print(column.capitalize() + ' - Missing Values: ' + str(sum(df[column].isnull())))


# In[25]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[26]:


sns.distplot(df[' Income '].dropna(),kde=False,color='darkred',bins=40)


# # Converting Categorical Features(One hot encoding)

# In[27]:


df = pd.get_dummies(df, columns = ['Education', 'Marital_Status']) 
print(df)


# In[28]:


df.dtypes


# In[29]:


print('Number of rows: ' + str(df.shape[0]) + '. Number of columns: ' + str(df.shape[1]))


# In[30]:


df.shape


# In[31]:


for column in df.columns:
    print(column,df[column].nunique())


# # Classification or Regression model 

# In[32]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['deposit']))]
print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# In[33]:


discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[34]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['deposit']]
print("Continuous feature Count {}".format(len(continuous_features)))


# In[35]:


# Discrete variables
discrete_columns = df.select_dtypes(include=['object']).columns

# Continuous variables
continuous_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("Discrete Variables:", discrete_columns)
print("Continuous Variables:", continuous_columns)


# In[36]:


#plot a univariate distribution of continues observations
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()


# In[37]:


## Checking for correlation
cor_mat=df.corr()
fig = plt.figure(figsize=(30,20))
sns.heatmap(cor_mat,annot=True)


# In[38]:


col_names = df.columns

col_names


# In[39]:


df.describe()


# In[40]:


#histogram to understand the distribution
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[41]:


#boxplot on numerical features to find outliers
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for numerical_feature in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(df[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# # Read class labels

# In[42]:


df1 = pd.read_csv('marketing_trn_class_labels.csv',header=None)


# In[43]:


df1


# In[44]:


df1.columns = ['Data_points', 'Class_lables']
df1.to_csv('marketing_trn_class_labels.csv', index=False)

print(df1)


# In[45]:


labels = df1['Class_lables']


# In[46]:


labels


# # Training and validfication the Data

# In[47]:


import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X = df
y = labels


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, classification_report


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)


# In[52]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix,\
precision_recall_curve, classification_report, accuracy_score, auc
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
#from imblearn.under_sampling import EditedNearestNeighbours
#from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
print('Libraries successfully imported.')


# In[53]:


# stratified cross validation object
skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=123)
# divide train_set in x_train and y_train


# # Decision Tree Classifier

# In[54]:


dt_classifier = DecisionTreeClassifier()

# Define the hyperparameter grid for tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=3, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best parameters
best_dt_classifier = DecisionTreeClassifier(**best_params)
best_dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and print the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# # Random Forest Classifier

# In[59]:


# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Define the hyperparameter grid for tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=3, scoring='f1_macro')
grid_search_rf.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_rf = grid_search_rf.best_params_
print("Best Hyperparameters for Random Forest:", best_params_rf)

# Train the model with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params_rf)
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy for Random Forest:", accuracy_rf)

# Print the Classification Report for Random Forest
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Compute and print the Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random Forest:")
print(conf_matrix_rf)


# # Logistic Regression classifier

# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have X_train and y_train from the previous example

# Create a Logistic Regression classifier
logreg_classifier = LogisticRegression()

# Define the hyperparameter grid for tuning
param_grid_logreg = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga']
}

# Use GridSearchCV for hyperparameter tuning
grid_search_logreg = GridSearchCV(estimator=logreg_classifier, param_grid=param_grid_logreg, cv=3, scoring='f1_macro')
grid_search_logreg.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_logreg = grid_search_logreg.best_params_
print("Best Hyperparameters for Logistic Regression:", best_params_logreg)

# Train the model with the best parameters
best_logreg_classifier = LogisticRegression(**best_params_logreg)
best_logreg_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logreg = best_logreg_classifier.predict(X_test)

# Evaluate the model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Accuracy for Logistic Regression:", accuracy_logreg)

# Print the Classification Report for Logistic Regression
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))

# Compute and print the Confusion Matrix for Logistic Regression
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_logreg)


# # K Neighbors Classifier

# In[61]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have X_train and y_train from the previous example

# Create a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()

# Define the hyperparameter grid for tuning
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Use GridSearchCV for hyperparameter tuning
grid_search_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, cv=3, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_knn = grid_search_knn.best_params_
print("Best Hyperparameters for k-Nearest Neighbors:", best_params_knn)

# Train the model with the best parameters
best_knn_classifier = KNeighborsClassifier(**best_params_knn)
best_knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = best_knn_classifier.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy for k-Nearest Neighbors:", accuracy_knn)

# Print the Classification Report for k-Nearest Neighbors
print("Classification Report for k-Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn))

# Compute and print the Confusion Matrix for k-Nearest Neighbors
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix for k-Nearest Neighbors:")
print(conf_matrix_knn)


# # Multinomial Naive Bayes classifer

# In[62]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have X_train and y_train from the previous example

# Create a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Define the hyperparameter grid for tuning
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

# Use GridSearchCV for hyperparameter tuning
grid_search_nb = GridSearchCV(estimator=nb_classifier, param_grid=param_grid_nb, cv=3, scoring='f1_macro')
grid_search_nb.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_nb = grid_search_nb.best_params_
print("Best Hyperparameters for Multinomial Naive Bayes:", best_params_nb)

# Train the model with the best parameters
best_nb_classifier = MultinomialNB(**best_params_nb)
best_nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = best_nb_classifier.predict(X_test)

# Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy for Multinomial Naive Bayes:", accuracy_nb)

# Print the Classification Report for Multinomial Naive Bayes
print("Classification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, y_pred_nb))

# Compute and print the Confusion Matrix for Multinomial Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(conf_matrix_nb)


# # Ada Boost Classifier

# In[63]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have X_train and y_train from the previous example

# Create an AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier()

# Define the hyperparameter grid for tuning
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}

# Use GridSearchCV for hyperparameter tuning
grid_search_adaboost = GridSearchCV(estimator=adaboost_classifier, param_grid=param_grid_adaboost, cv=3, scoring='f1_macro')
grid_search_adaboost.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_adaboost = grid_search_adaboost.best_params_
print("Best Hyperparameters for AdaBoost:", best_params_adaboost)

# Train the model with the best parameters
best_adaboost_classifier = AdaBoostClassifier(**best_params_adaboost)
best_adaboost_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_adaboost = best_adaboost_classifier.predict(X_test)

# Evaluate the model
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print("Accuracy for AdaBoost:", accuracy_adaboost)

# Print the Classification Report for AdaBoost
print("Classification Report for AdaBoost:")
print(classification_report(y_test, y_pred_adaboost))

# Compute and print the Confusion Matrix for AdaBoost
conf_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)
print("Confusion Matrix for AdaBoost:")
print(conf_matrix_adaboost)


# # Support Vector Machine classifier

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have X_train and y_train from the previous example

# Create a Support Vector Machine classifier
svm_classifier = SVC()

# Define the hyperparameter grid for tuning
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV for hyperparameter tuning
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, cv=3, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params_svm = grid_search_svm.best_params_
print("Best Hyperparameters for Support Vector Machine:", best_params_svm)

# Train the model with the best parameters
best_svm_classifier = SVC(**best_params_svm)
best_svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = best_svm_classifier.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy for Support Vector Machine:", accuracy_svm)

# Print the Classification Report for Support Vector Machine
print("Classification Report for Support Vector Machine:")
print(classification_report(y_test, y_pred_svm))

# Compute and print the Confusion Matrix for Support Vector Machine
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix for Support Vector Machine:")
print(conf_matrix_svm)


# In[ ]:




