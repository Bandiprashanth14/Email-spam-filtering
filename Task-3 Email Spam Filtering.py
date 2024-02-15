#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Email spam filtering is a technique used to automatically identify and

categorize unwanted or unsolicited emails, commonly known as spam, from 
legitimate and

relevant emails in a user's inbox. The goal of email spam filtering is to 
reduce the clutter in users' email accounts,

enhance security by minimizing the risk of phishing and malware attacks, and 
improve the overall email experience.

'''


# In[ ]:


'''
panda,numpy,matplotlib,seaborn,sklearn are the basic libraries used in the

email spam filtering

natural language tool kit used to study the data which means a mail

and visualized the data in the different graphical form(pictorial representation

'''


# In[3]:


import os
os.getcwd()


# In[4]:


# Import Necessary Libraries

import pandas as pd
import numpy as np

# Libraries for visualisation

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve,roc_auc_score
import nltk
from nltk.corpus import stopwords
from collections import Counter


# In[5]:


# Download the stopwords dataset

nltk.download('stopwords')


# **Encoding the Given Data**

# In[9]:


df=pd.read_csv('C:\\Users\\gouthami\\Downloads\\Techno Hacks\\Email_Spam\\spam.csv',encoding='latin-1')
df.head()


# In[10]:


df.columns


# In[11]:


df.info


# In[12]:


df.head()


# In[13]:


df.tail()


# **dropping the Unnecessary columns**

# In[15]:


columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
df.drop(columns=columns_to_drop, inplace=True)


# In[16]:


df


# In[17]:


# After removing the unwanted Columns

df.columns


# **Renaming the columns which we have**

# In[18]:


# Rename the columns "v1 and v2" to new names

new_column_names = {"v1":"Category","v2":"Message-in-email"}
df.rename(columns = new_column_names,inplace = True)


# In[19]:


df


# **Replacing the null values in the Data Frame**

# In[20]:


data = df.where((pd.notnull(df)),' ')


# In[21]:


df.head(20)


# In[22]:


data.describe()


# In[23]:


data.info()


# In[24]:


data.shape


# In[26]:


# Convert the "Category" column values to numerical representation (0 for"spam" and 1 for "ham")

data.loc[data["Category"] == "spam", "Category"] = 0
data.loc[data["Category"] == "ham", "Category"] = 1


# In[27]:


# Separate the feature (message) and target (category) data
    
X = data["Message-in-email"]
Y = data["Category"]


# In[28]:


print(X)


# In[30]:


print(Y)


# In[31]:


# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 3)


# In[32]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[33]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[34]:


# Create a TF-IDF vectorizer to convert text messages into numerical features

feature_extraction = TfidfVectorizer(min_df=1, stop_words="english",lowercase=True)


# In[35]:


# Convert the training and testing text messages into numerical features using TF-IDF

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[36]:


# Convert the target values to integers (0 and 1)

Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")


# In[37]:


print(X_train)


# In[38]:


print(X_train_features)


# **2. Logical Regression**

# In[39]:


# Create a logistic regression model and train it on the training data

model = LogisticRegression()
model.fit(X_train_features, Y_train)


# In[40]:


# Make predictions on the training data and calculate the accuracy
    
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[41]:


print("Accuracy on training data:",accuracy_on_training_data)


# In[42]:


# Make predictions on the test data and calculate the accuracy
    
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)


# In[43]:


print("Accuracy on test data:",accuracy_on_test_data)


# In[44]:


# Test the model with some custom email messages

input_your_mail = ["Congratulations! You have won a free vacation to an exotic destination. Click the link to claim your prize now!"]

input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

# Print the prediction result

if (prediction)[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")


# In[46]:


# Test the model with some custom email messages

input_your_mail = ["Meeting reminder: Tomorrow, 10 AM, conference room. See you there!"]
                   
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)
                   
# Print the prediction result
                   
if (prediction)[0] == 1:
 print("Ham Mail")
else:
 print("Spam Mail")


# In[47]:


# Data visualization- Distribution of Spam and Ham Emails

spam_count = data[data['Category'] == 0].shape[0]
ham_count = data[data['Category'] == 1].shape[0]

plt.bar(['Spam', 'Ham'], [spam_count, ham_count])
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Spam and Ham Emails')
plt.show()


# In[48]:


# Data visualization- Confusion Matrix

cm = confusion_matrix(Y_test, prediction_on_test_data)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[49]:


# Data visualization- ROC Curve

probabilities = model.predict_proba(X_test_features)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)
roc_auc = roc_auc_score(Y_test, probabilities)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[50]:


# Data visualization- Top 10 Most Common Words in Spam Emails

stop_words = set(stopwords.words('english'))
spam_words = " ".join(data[data['Category'] == 0]['Message-in-email']).split()
ham_words = " ".join(data[data['Category'] == 1]['Message-in-email']).split()

spam_word_freq = Counter([word.lower() for word in spam_words if word.lower() not in stop_words and word.isalpha()])
plt.figure(figsize=(10, 6))
plt.bar(*zip(*spam_word_freq.most_common(10)), color='g')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Spam Emails')
plt.xticks(rotation=45)
plt.show()


# In[51]:


# Data visualization- Top 10 Most Common Words in Ham Emails

ham_word_freq = Counter([word.lower() for word in ham_words if word.lower() not in stop_words and word.isalpha()])

plt.figure(figsize=(10, 6))
plt.bar(*zip(*ham_word_freq.most_common(10)), color='b')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Ham Emails')
plt.xticks(rotation=45)
plt.show()


# In[52]:


from sklearn.preprocessing import LabelEncoder


# In[53]:


le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
df["Category"].value_counts()


# In[54]:


sns.countplot(x='Category',data=df)
plt.show()


# In[55]:


# by above 1-is spam and the 0 is ham
# pie chart representation of the mails-spam and ham

plt.pie(df["Category"].value_counts(),autopct = "%.2f", labels=['ham','spam'])
plt.show()


# In[ ]:




