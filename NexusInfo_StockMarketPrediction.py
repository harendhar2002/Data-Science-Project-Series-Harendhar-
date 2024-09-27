#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('D:/Extra Learning/Internships/Nexus Info/Project 1/infolimpioavanzadoTarget.csv',header=0)
df.head()


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


stock=df[['date','open','high','low','close','adjclose','volume']]


# In[6]:


stock.describe()


# In[7]:


import matplotlib.pyplot as plt

# Plot closing prices
plt.figure(figsize=(10,6))
plt.plot(stock['close'],label='Closing Price')
plt.title('Apple Stock Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# # Method 1 of Stock Prediction by Classification Model

# # EDA

# In[8]:


stock=stock.copy()


# In[9]:


stock['tomorrow']=stock['close'].shift(-1)


# In[10]:


stock['Target']=(stock['tomorrow']>stock['close']).astype(int)
stock.head()


# # Train the Model

# In[11]:


from sklearn.ensemble import RandomForestClassifier


# In[12]:


rf_model=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)


# In[13]:


train=stock.iloc[:-100]
test=stock.iloc[-100:]


# In[14]:


predictors=["open","high","low","close","volume"]
rf_model.fit(train[predictors],train["Target"])


# In[15]:


from sklearn.metrics import precision_score
preds=rf_model.predict(test[predictors])


# In[16]:


#preds will be array so to convert into series 
preds=pd.Series(preds,index=test.index)


# In[17]:


preds


# In[40]:


precision_score(test["Target"],preds)


# In[41]:


from sklearn.metrics import classification_report


# In[44]:


re=classification_report(test["Target"],preds)
print(re)


# In[19]:


combined=pd.concat([test["Target"],preds],axis=1)


# In[20]:


combined.plot()


# In[21]:


stock.reset_index(drop=True,inplace=True)


# # Method 2 of Stock Prediction by Regression Model

# ## To Perform Stock Market Prediction
# 1) Moving Average
# 2) Lagged Price

# In[27]:


# Moving Average at  25 days,50days and 200 days
stock.loc[:,'Moving_Avg25']=stock.loc[:,'close'].rolling(window=25).mean()
stock.loc[:,'Moving_Avg50']=stock.loc[:,'close'].rolling(window=50).mean()
stock.loc[:,'Moving_Avg200']=stock.loc[:,'close'].rolling(window=200).mean()


# In[28]:


# adding new column with yesterday 'close value' data
stock.loc[:,'Yesterday']=stock.loc[:,'close'].shift(1)


# In[29]:


stock.dropna(inplace=True)


# In[30]:


x=stock[["Moving_Avg25","Moving_Avg50","Moving_Avg200","Yesterday"]]
y=stock["close"]


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[33]:


rf_model3=RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=1)
rf_model3.fit(x_train,y_train)


# In[34]:


y_test_pred=rf_model3.predict(x_test)


# In[35]:


y_test_pred=pd.Series(y_test_pred)


# In[36]:


mean_squared_error(y_test,y_test_pred)


# ### Plotting the Trained Model

# In[37]:


plt.figure(figsize=(15,8))
plt.title("Actual Stock Prices")
plt.plot(y_test.index,y_test,label="Actual Price",color='b')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()


# In[38]:


plt.figure(figsize=(15,8))
plt.title("Predicted Stock Prices")
plt.plot(y_test.index,y_test_pred,label="Predicted Price",color='r')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()


# In[39]:


plt.figure(figsize=(15,8))
plt.title("Actual Prices vs Predicted Prices")
plt.plot(y_test.index,y_test,label="Actual Price",color='b')
plt.plot(y_test.index,y_test_pred,label="Predicted Price",color='r')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()

