#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bc=pd.read_csv('D:/Extra Learning/Internships/Nexus Info/Project 2/data.csv',header=0)
bc.head()


# In[3]:


bc.describe().T


# In[4]:


bc.diagnosis.unique()


# In[5]:


bc['diagnosis'].value_counts()


# In[6]:


sns.countplot(bc['diagnosis'],palette='husl')


# In[7]:


bc.drop('id',axis=1,inplace=True)
bc.drop('Unnamed: 32',axis=1,inplace=True)


# In[8]:


# mapping maligent as 1 and benign as 0 
bc['diagnosis']=bc['diagnosis'].map({'M':1,'B':0})
bc.head()


# In[9]:


#plotting correlation graph
plt.figure(figsize=(20,20))
sns.heatmap(bc.corr(),annot=True)
plt.show()


# In[10]:


cols=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
      'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']
sns.pairplot(data=bc[cols],hue='diagnosis',palette='rocket')
plt.show()


# In[11]:


# dropping unneccesary columns
col2=['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
     'concave points_worst','symmetry_worst','fractal_dimension_worst']

bc=bc.drop(col2,axis=1)


# In[12]:


col3=['perimeter_mean','perimeter_se','area_mean','area_se','concavity_mean','concavity_se',
      'concave points_mean','concave points_se']
bc=bc.drop(col3,axis=1)


# In[13]:


bc.head()


# ### Training thr Model

# In[14]:


x=bc.drop(['diagnosis'],axis=1)
y=bc['diagnosis']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# ### Feature Scaling

# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


x_trains=StandardScaler().fit_transform(x_train)
x_tests=StandardScaler().fit_transform(x_test)


# In[19]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[64]:


svm_model=SVC()
svm_model.fit(x_trains,y_train)


# In[65]:


y_pred=svm_model.predict(x_tests)


# In[66]:


accuracy_score(y_test,y_pred)


# In[67]:


confusion_matrix(y_test,y_pred)


# In[68]:


report=classification_report(y_test,y_pred)
print(report)


# ### Creating Best Model with SVM for accuracy

# In[25]:


from sklearn.model_selection import GridSearchCV


# In[55]:


param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
}


# In[57]:


grid_search=GridSearchCV(SVC(),param_grid,n_jobs=-1,cv=5,refit=True,verbose=2)
grid_search.fit(x_trains,y_train)


# In[58]:


bestSVC_model=grid_search.best_estimator_
y_pred2=bestSVC_model.predict(x_tests)


# In[59]:


report2=classification_report(y_test,y_pred2)
print(report2)


# In[60]:


grid_search.best_params_


# In[63]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




