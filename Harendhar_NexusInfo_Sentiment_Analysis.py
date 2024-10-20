#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Imports for text cleaning
import string
import re
import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer


# In[3]:


# Imports for preprocessing, modeling and evaluation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.tree import plot_tree

from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier
from xgboost import plot_importance

import time


# In[4]:


df_train=pd.read_csv('D:/Extra Learning/Internships/Nexus Info/train.csv',header=0)
df_train.head()


# In[5]:


df_test=pd.read_csv('D:/Extra Learning/Internships/Nexus Info/test.csv',header=0)
df_test.head()


# In[6]:


print(df_train.shape,df_test.shape)


# In[7]:


print(df_train.info(),df_test.info())


# In[8]:


print(df_train.isnull().sum(),df_test.isnull().sum())


# In[9]:


#Remove rows with missing data in both datasets
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)


# In[10]:


print(df_train.isnull().sum(),df_test.isnull().sum())


# In[11]:


#Rename columns as needed
df_train.columns= df_train.columns.str.lower()
df_train=df_train.rename(columns={'textid':'text_id','time of tweet':'time_of_tweet','age of user':'age_of_user',
                                  'population -2020':'population_2020','land area (km�)':'land_area_km2',
                                  'density (p/km�)':'density_p_km2'})
df_test.columns= df_test.columns.str.lower()
df_test=df_test.rename(columns={'textid':'text_id','time of tweet':'time_of_tweet','age of user':'age_of_user',
                                'population -2020':'population_2020','land area (km�)':'land_area_km2',
                                'density (p/km�)':'density_p_km2'})


# In[12]:


df_train.head()


# In[13]:


# Check if the distribution of sentiment labels in the dataset is imbalanced.
df_train['sentiment'].value_counts()


# In[14]:


# Visualizing tweet time distribution with a histogram to determine its influence on sentiment
sns.countplot(data=df_train,x='time_of_tweet',hue='sentiment')
plt.title("Frequency of sentiments per time of tweet")
plt.xlabel("Time of tweet")
plt.ylabel("Frequency of sentiment")


# In[15]:


# Visualizing age-of-user distribution with a histogram to determine its influence on sentiment
sns.countplot(data=df_train,x='age_of_user',hue='sentiment')
plt.title("Frequency of sentiments per user age")
plt.xlabel("Age of user")
plt.ylabel("Frequency of sentiment")


# In[16]:


# Keep only the 'text' and 'sentiment' columns for further analysis.
df1_train=df_train[['text','sentiment']].copy()


# In[17]:


# Perform cleaning and preprocessing on the 'text' column
def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Remove newlines
    text = re.sub(r'\n', '', text)
    # Remove alphanumeric words (words containing digits)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove remaining non-alphabetic characters (except spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Normalize repeated characters (e.g., "soooo" -> "so")
    text = re.sub(r'(.)\1+', r'\1\1', text)
  
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stem words using LancasterStemmer
    stemmer = LancasterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join words back into a single string
    text = ' '.join(words)
    
    return text

# Ensure the 'text' column is of string type
df1_train['text'] = df1_train['text'].astype(str)
df_test['text'] = df_test['text'].astype(str)

# Apply the clean_text function and assign it back to the DataFrame
df1_train['text'] = df1_train['text'].apply(clean_text)
df_test['text'] = df_test['text'].apply(clean_text)


# In[18]:


# Display the first 10 rows of the training dataset.
df1_train.head(10)


# In[19]:


# Select features for training and testing datasets
X_train = df1_train['text']
X_test = df_test['text']
y_train = df1_train['sentiment']
y_test = df_test['sentiment']


# In[20]:


# Apply vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
XV_train = vectorizer.fit_transform(X_train)
XV_test = vectorizer.transform(X_test)


# ### Implementation and Analysis of Machine Learning Models
# **We are going to use:**
# 
# - Logistic Regression
# - Naive Bayes
# - Decision Tree Model
# - Random Forest Model
# - XGBoost Model

# ### Logistic Regression

# In[21]:


# Logistic Regression model 
print("Training....")
t0=time.time()
clf = LogisticRegression(max_iter = 300).fit(XV_train,y_train)
train_time = time.time()-t0
print(f"train time: {train_time:.3}s")


# In[22]:


# Predict on test set.
t0=time.time()
y_pred = clf.predict(XV_test)
predict_time = time.time()-t0
print(f"predict time: {predict_time:.3}s")


# In[23]:


# Analyze the results
def analyze_results(y_test,y_pred):
    # Get the metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision =  metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", "%.6f" % accuracy)
    print("Precision:", "%.6f" % precision)
    print("Recall:", "%.6f" %  recall)
    print("F1 Score:", "%.6f" %  f1_score)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    cm = metrics.confusion_matrix(y_test,y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    return accuracy, precision, recall, f1_score


# In[29]:


# Print results
logistic_regression_accuracy, logistic_regression_precision, logistic_regression_recall,logistic_regression_f1_score = analyze_results(y_test,y_pred)


# ### Naive Bayes

# In[27]:


# Naive Bayes model
nb = naive_bayes.MultinomialNB()
print("Training....")
t0=time.time()
nb.fit(XV_train, y_train)
train_time = time.time()-t0
print(f"train time: {train_time:.3}s")


# In[28]:


# Predict on test set.
t0=time.time()
y_pred = nb.predict(XV_test)
predict_time = time.time()-t0
print(f"predict time: {predict_time:.3}s")


# In[31]:


# Print results
naive_bayes_accuracy, naive_bayes_precision, naive_bayes_recall, naive_bayes_f1_score = analyze_results(y_test,y_pred)


# ### Decision Tree

# In[32]:


# Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=0)
print("Training....")
t0=time.time()
decision_tree.fit(XV_train, y_train)
train_time = time.time()-t0
print(f"train time: {train_time:.3}s")


# In[33]:


# Predict on test set.
t0=time.time()
y_pred = decision_tree.predict(XV_test)
predict_time = time.time()-t0
print(f"predict time: {predict_time:.3}s")


# In[34]:


# Print results
decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_f1_score = analyze_results(y_test,y_pred)


# ### Random Forest

# In[35]:


# Random Forest model
cv_params = {'n_estimators' : [50,100], 
              'max_depth' : [10,50],        
              'min_samples_leaf' : [0.5,1], 
              'min_samples_split' : [0.001, 0.01],
              'max_features' : ["sqrt"], 
              'max_samples' : [.5,.9]}
rf = RandomForestClassifier(random_state=0)
rf_val = GridSearchCV(rf, cv_params,  refit='f1', n_jobs = -1, verbose = 1)

print("Training....")
t0=time.time()
rf_val.fit(XV_train, y_train)
train_time = time.time()-t0
print(f"train time: {train_time:.3}s")


# In[36]:


# Obtain the best parameter set identified by GridSearchCV
rf_val.best_params_


# In[37]:


# Use optimal parameters from GridSearchCV.
rf_opt = RandomForestClassifier(n_estimators = 50, max_depth = 50, 
                                min_samples_leaf = 1, min_samples_split = 0.001,
                                max_features="sqrt", max_samples = 0.9, random_state = 0)


# In[38]:


# Fit the optimal model.
rf_opt.fit(XV_train, y_train)


# In[39]:


# Predict on test set.
y_pred = rf_opt.predict(XV_test)


# In[44]:


# Print results
random_forest_accuracy, random_forest_precision, random_forest_recall,random_forest_f1_score = analyze_results(y_test,y_pred)


# ### XGBoost

# In[43]:


#XGBoost model
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
cv_params = {'max_depth': [2, 6],
               'min_child_weight': [1, 5],
              'learning_rate': [0.01, 0.2, 0.3],
               'n_estimators': [5,10,15],
               'subsample': [0.7],
               'colsample_bytree': [0.7]
              }
xgb = XGBClassifier(random_state=0, max_depth=10, n_estimators=100,learning_rate=0.01)
xgb_cv = GridSearchCV(xgb , cv_params,  refit='f1', n_jobs = -1, cv = 5)

print("Training....")
t0=time.time()
xgb_cv.fit(XV_train, y_train_encoded)
train_time = time.time()-t0
print(f"train time: {train_time:.3}s")


# In[45]:


# Obtain the best parameter set identified by GridSearchCV
xgb_cv.best_params_


# In[46]:


# Use optimal parameters from GridSearchCV.
xgb_opt = XGBClassifier(n_estimators = 15, max_depth = 6, min_child_weight = 1,
                          learning_rate = 0.3, colsample_bytree= 0.7, subsample = 0.7)


# In[47]:


# Fit the optimal model.
xgb_opt.fit(XV_train, y_train_encoded)


# In[48]:


# Predict on test set.
y_pred = xgb_cv.predict(XV_test)


# In[49]:


# Print results
y_test_encoded = label_encoder.fit_transform(y_test)
xgboost_accuracy, xgboost_precision, xgboost_recall,xgboost_f1_score = analyze_results(y_test_encoded,y_pred)


# ## Summary of Model Results and Conclusion

# In[50]:


# Generate report
table = pd.DataFrame({'Model': ["Logistic Regression","Naive Bayes","Decision Tree", "Random Forest","XGBoost model"],
                        'F1':  [logistic_regression_f1_score, naive_bayes_f1_score, decision_tree_f1_score, random_forest_f1_score, xgboost_f1_score],
                        'Recall':  [logistic_regression_recall, naive_bayes_recall, decision_tree_recall, random_forest_recall, xgboost_recall],
                        'Precision': [logistic_regression_precision, naive_bayes_precision, decision_tree_precision, random_forest_precision, xgboost_precision],
                        'Accuracy': [logistic_regression_accuracy, naive_bayes_accuracy, decision_tree_accuracy, random_forest_accuracy, xgboost_accuracy],
                      }
                    )
table


# In[51]:


# Export results to CSV
table.to_csv("Report_text_after_stematization.csv")


# ## Conclusion: 
# 
# The best results are obtained from the Logistic Regression model, followed closely by the XGBoost model.

# In[ ]:




