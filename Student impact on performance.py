#!/usr/bin/env python
# coding: utf-8

# # Student Impact on Performance Analysis

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


# ## Reading the file

# In[3]:


data = pd.read_excel('D:/Data Science/07July2023/Questionaire_Data.xlsx', sheet_name = 'Processed Data')


# In[4]:


data.head(5)


# ## Basic Information of Data

# In[8]:


data.info()


# ## Checking Duplicates

# In[29]:


data.duplicated().sum()


# ## Checking null values

# In[30]:


data.isnull().sum()


# ## Unique Value in Columns

# In[4]:


print(data['College Name'].value_counts())
print(data['Family Income_monthly'].value_counts())
print(data['Academic Level'].value_counts())
print(data['Online Attendance Before Covid'].value_counts())
print(data['Learning Device'].value_counts())
print(data['Internet Type'].value_counts())
print(data['Weekly_40_hrs_study'].value_counts())
print(data['Home Internet Access'].value_counts())
print(data['No Offline Lectures'].value_counts())
print(data['Online Topic Clarification'].value_counts())
print(data['Unique Online Methods'].value_counts())
print(data['Avg Time Spend'].value_counts())
print(data['Online Methods Helpful'].value_counts())
print(data['Teachers Helpful Online'].value_counts())
print(data['Online Environment Preferred'].value_counts())
print(data['College Environment Preferred'].value_counts())



# # Exploratory Data Analysis

# In[5]:


data['College Name'].value_counts().head(5).plot(kind='barh',title='Top 5 Colleges',color='lightblue')


# ## Top 5 colleges where student have enrolled

# In[39]:


sns.countplot(x='Family Income_monthly', data = data, palette='coolwarm')


# ## Family Monthly Income

# In[40]:


sns.displot(x='Academic Level',data=data)


# ## Academic Level of Students

# In[43]:


data['Online Attendance Before Covid'].value_counts().plot(kind='pie',autopct='%0.0f%%')


# ## Online Attendance before Covid is 57%

# In[49]:


data['Learning Device'].value_counts().head().plot(kind='barh',title='Top Learning Devices',color='lightgrey')


# ## This clearly shows that smartphone is most likely used by students

# In[50]:


sns.displot(x='Internet Type',data=data)


# ## This shows that mobile internet is mostly used by student.

# In[56]:


plt.figure(figsize=(30,15))
sns.countplot(x='Online Platforms',data=data)


# ## This shows that Zoom is mostly used online platform by the students

# In[63]:


plt.figure(figsize=(12,8))
sns.countplot(x='Online Teaching Method',data=data)


# ## Online Teaching method preferred is Live Lecture

# In[61]:


data['Weekly_40_hrs_study'].value_counts().plot(kind='pie',autopct='%0.0f%%')


# ## 69% of students are dedicating for weekly 40 hrs study

# In[68]:


plt.figure(figsize=(8,6))
sns.countplot(x='Avg Time Spend',data=data)


# ## Avg time spend by students are either 2-3 hrs or 3-5 hrs

# In[66]:


plt.figure(figsize=(8,6))
sns.countplot(x='Hectic Distance Learning',data=data)


# ## More then 140 students says its slightly hectic distance learning

# In[70]:


plt.figure(figsize=(10,6))
sns.countplot(x='Teachers Helpful Online',data=data)


# ## More then 150 students says teachers are helpful online

# In[75]:


data['Learn from Home'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['Lightgrey','LightBlue'])


# ## 62% of students wants to learn from home

# In[81]:


plt.figure(figsize=(10,6))
sns.countplot(x='Traditional Teaching Method',data=data)


# ## More then 140 students agree with Traditional teaching method

# In[82]:


plt.figure(figsize=(10,6))
sns.countplot(x='Online Methods Helpful',data=data)


# ## More then 80 students agree with Online method and 100+ are neutral

# In[83]:


data['Future Preferred Learning Mode'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['grey','pink'])


# ## 53% of stdents says offline learning method for future

# In[18]:


plt.figure(figsize=(10,20))
pd.crosstab(data['Future Preferred Learning Mode'],data['Isolation Increased']).plot(kind='bar')


# ## More the 120 students says yes isolation increases be it offline/online

# In[8]:


pd.crosstab(data['Future Preferred Learning Mode'],data['Secured Learning']).plot(kind='bar')


# ## More then 140 students says online gives secured learning 

# In[9]:


pd.crosstab(data['Future Preferred Learning Mode'],data['Time Effectivity']).plot(kind='bar')


# ## More then 140 students agrees online learning mode is time effective

# In[11]:


pd.crosstab(data['Future Preferred Learning Mode'],data['Concentration lack ']).plot(kind='bar')


# ## with Offline more then 140 students say yes the concentration lacks 
# ## with online 100+ students say yes for conentartion lacks

# In[84]:


data['Electronic gadget'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['lightgrey','blue'])


# ## 82% of students agrees that electronic gadget improved technical skills in online learning methods

# In[10]:


data['Online Interactive Sessions'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['lightgrey','brown'])


# ## 69% of students agrees for Online Interactive Sessions

# In[11]:


data['Missing Pratical Knowledge'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['lightgrey','lightgreen'])


# ## 79% of students says they are missing practical knowledge due to online classes

# In[16]:


data['Online Lecture Useful'].value_counts().plot(kind='pie',autopct='%0.0f%%',colors=['grey','green'])


# ## 83% of students agrees with online lectures are useful

# In[18]:


pd.crosstab(data['Address'],data['Online Environment Preferred']).plot(kind='bar')


# ## More then 120 students fom rural areas says no to online preferrence where as in urban we see mixed reactions

# In[8]:


pd.crosstab(data['Online Environment Preferred'],data['College Environment Preferred']).plot(kind='bar',title='College / Online Prefference')


# ## The above graphs shows that college environment is most preffered then online.

# # Conclusion
# 
# ## There is an impact on student performance through online learning
# ## 1) The students gets benefits in many ways like enhancing and maximizing thier learning independence.
# ## 2) Students were more engaged in the learning process than in conventional teaching.
# ## 3) Teachers are very helpful online.
# ## 4) Students agrees that online lectures are more useful
# ## 5) The electronic gadgets helps in technical skills in online learning
# 
# ## There is an negative impact as well in online learning
# ## 1) Some students like offline learning, Tradtional way of teaching
# ## 2) Missing on practical Knowlegdge
# ## 3) Face to Face Intercation missing
# ## 4) In rural areas still due to internet connectivity issues students preferred offline learning method.
# ## 5) College environment is preferred by the students rather the online environment.
# 
# 

# # Applying Decision Tree Classification Method

# In[9]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn import tree


# ## Since all the data types are object we need to use label encoder to convert into numerical

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


le = LabelEncoder()


# In[13]:


data.columns


# In[14]:


data['Address'] = le.fit_transform(data['Address'])
data['Family Type'] = le.fit_transform(data['Family Type'])
data['Fathers Education'] = le.fit_transform(data['Fathers Education'])
data['Mothers Education'] = le.fit_transform(data['Mothers Education'])
data['Family Income_monthly'] = le.fit_transform(data['Family Income_monthly'])
data['Academic Level'] = le.fit_transform(data['Academic Level'])
data['Online Attendance Before Covid'] = le.fit_transform(data['Online Attendance Before Covid'])
data['Learning Device'] = le.fit_transform(data['Learning Device'])
data['Internet Type'] = le.fit_transform(data['Internet Type'])
data['Home Internet Access'] = le.fit_transform(data['Home Internet Access'])
data['Online Platforms'] = le.fit_transform(data['Online Platforms'])
data['Online Teaching Method'] = le.fit_transform(data['Online Teaching Method'])
data['Comfortable Technology '] = le.fit_transform(data['Comfortable Technology '])
data['Internet Access'] = le.fit_transform(data['Internet Access'])
data['Weekly_40_hrs_study'] = le.fit_transform(data['Weekly_40_hrs_study'])
data['Avg Time Spend'] = le.fit_transform(data['Avg Time Spend'])
data['Downloading Knowledge'] = le.fit_transform(data['Downloading Knowledge'])
data['Self Motivated'] = le.fit_transform(data['Self Motivated'])
data['Hours Spend'] = le.fit_transform(data['Hours Spend'])
data['Email_WebBrowser'] = le.fit_transform(data['Email_WebBrowser'])
data['Teacher Meet'] = le.fit_transform(data['Teacher Meet'])
data['No Offline Lectures'] = le.fit_transform(data['No Offline Lectures'])
data['Online Topic Clarification'] = le.fit_transform(data['Online Topic Clarification'])
data['Unique Online Methods'] = le.fit_transform(data['Unique Online Methods'])
data['Learn from Home'] = le.fit_transform(data['Learn from Home'])
data['Hectic Distance Learning'] = le.fit_transform(data['Hectic Distance Learning'])
data['Teachers Helpful Online'] = le.fit_transform(data['Teachers Helpful Online'])
data['Syallbus Coverage '] = le.fit_transform(data['Syallbus Coverage '])
data['Traditional Teaching Method'] = le.fit_transform(data['Traditional Teaching Method'])
data['Online Methods Helpful'] = le.fit_transform(data['Online Methods Helpful'])
data['Isolation Increased'] = le.fit_transform(data['Isolation Increased'])
data['Secured Learning'] = le.fit_transform(data['Secured Learning'])
data['Time Effectivity'] = le.fit_transform(data['Time Effectivity'])
data['Online Learning Continuation'] = le.fit_transform(data['Online Learning Continuation'])
data['Quick Response '] = le.fit_transform(data['Quick Response '])
data['Electronic gadget'] = le.fit_transform(data['Electronic gadget'])
data['Online Interactive Sessions'] = le.fit_transform(data['Online Interactive Sessions'])
data['Missing Pratical Knowledge'] = le.fit_transform(data['Missing Pratical Knowledge'])
data['Concentration lack '] = le.fit_transform(data['Concentration lack '])
data['Online Lecture Useful'] = le.fit_transform(data['Online Lecture Useful'])
data['Online Environment Preferred'] = le.fit_transform(data['Online Environment Preferred'])
data['College Environment Preferred'] = le.fit_transform(data['College Environment Preferred'])


# In[15]:


data.head(5)


# In[18]:


x = data.iloc[:,4:46]
x


# In[20]:


y = data['Future Preferred Learning Mode']
y


# In[22]:


model = DecisionTreeClassifier(criterion = 'gini')
clf = model.fit(x,y)


# In[23]:


pred = clf.predict(x)
pred


# In[24]:


result = confusion_matrix(y,pred)
result


# In[25]:


ConfusionMatrixDisplay(result, display_labels=clf.classes_).plot()
plt.show()


# In[40]:


plt.figure(figsize=(20,25))
feat_imp = pd.Series(clf.feature_importances_, index=x.columns)
feat_imp.plot(kind='barh')


# In[30]:


accuracy_score(y,pred)*100


# In[35]:


plt.figure(figsize=(50,30))
tree.plot_tree(clf)


# ## The Model is Prediciting 100% accuracy
