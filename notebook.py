#!/usr/bin/env python
# coding: utf-8

# # Stats Smackdown: ML Predictions in MMA

# Load Modules, Dataset, and Data Frame

# In[2]:


import pandas as pan
import numpy as num
import matplotlib.pyplot as mp

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud


# In[3]:


dfr = pan.read_csv('ufc-fighters-statistics.csv')


# Inspect Data Frame

# In[4]:


print(dfr.shape)
dfr.head(25)


# In[5]:


dfr.dtypes


# Prepare Data and String Operations

# In[6]:


dfr['name'].str.replace(' ', '_').str.lower()


# In[7]:


strings = list(dfr.dtypes[dfr.dtypes == 'object'].index)
strings


# In[8]:


dfr.index


# Delete Column

# In[9]:


unwanted_column = 'nickname'

if unwanted_column in dfr.columns:
    # Drop the unwanted column
    dfr = dfr.drop(unwanted_column, axis=1)
    print(f"\n'{unwanted_column}' column is not in the data frame.")
else:
    print(f"\n'{unwanted_column}' column has been successfully dropped.")

# Print remaining column names
print("\nColumns after deletion:")
print(dfr.columns)


# Discover Missing Values

# In[10]:


dfr.isnull()


# In[11]:


dfr.isnull().sum()


# In[12]:


dfr_cleaned = dfr.dropna(subset=['height_cm', 'weight_in_kg', 'reach_in_cm', 'stance', 'date_of_birth'])


# Summarize Findings

# In[13]:


dfr.describe().round(2)


# In[14]:


dfr.nunique()


# ## Extensive EDA

# Range of Values

# In[15]:


numb_cols = dfr.select_dtypes(include=['int64', 'float64']).columns

col_ranges = dfr[numb_cols].agg(['min', 'max', 'mean', 'std']).transpose()

print("Numeric Column Ranges:")
print(col_ranges)


# * 'wins': The total number of victories in the fighter's career.
# * 'losses': The total number of losses in the fighter's career.
# * 'draws': The total number of draws or ties in the fighter's career.
# * 'height_cm': The fighter's height in centimeters.
# * 'weight_in_kg': The fighter's weight in kilograms.
# * 'reach_in_cm': The fighter's reach, measured in centimeters.
# * 'stance': The preferred fighting stance of the fighter (orthodox, southpaw, etc.).
# * 'date_of_birth': The fighter's date of birth.
# * 'significant_strikes_landed_per_minute': The average number of significant strikes landed by the fighter per minute.
# * 'significant_striking_accuracy': The percentage of significant strikes that the fighter lands successfully.
# * 'significant_strikes_absorbed_per_minute': The average number of significant strikes absorbed by the fighter per minute.
# * 'significant_strike_defence': The percentage of significant strikes that the fighter successfully defends against.
# * 'average_takedowns_landed_per_15_minutes': The average number of successful takedowns by the fighter per 15 minutes.
# * 'takedown_accuracy': The percentage of takedown attempts by the fighter that are successful.
# * 'takedown_defense': The percentage of takedown attempts against the fighter that are defended successfully.
# * 'average_submissions_attempted_per_15_minutes': The average number of submission attempts by the fighter per 15 minutes.

# Analysis of Target Variable

# In[16]:


mp.figure(figsize=(12, 7))
mp.hist(dfr['wins'], bins='auto', color='green', edgecolor='black')
mp.title('Distribution of Wins')
mp.xlabel('Wins')
mp.ylabel('Count')
mp.show()


# In[17]:


filtered_data = dfr[dfr['wins'] < 50]

mp.figure(figsize=(10, 6))
mp.hist(filtered_data['wins'], bins='auto', color='skyblue', edgecolor='black')
mp.title('Distribution of Wins (Less Than 50)')
mp.xlabel('Wins')
mp.ylabel('Count')
mp.show()


# In[18]:


num.log1p([0, 1, 10, 20, 30])


# In[19]:


num.log([0 + 1, 1+ 1, 10 + 1, 20 + 1, 30])


# In[20]:


win_logs = num.log1p(dfr.wins)


# In[21]:


mp.hist(win_logs, bins=30)


# Validation Framework & Train Test Split

# In[22]:


full_train, test_df = train_test_split(dfr, test_size=0.1, random_state=2)
train_df, val_df = train_test_split(full_train, test_size=0.50, random_state=2)


# In[23]:


len(train_df), len(val_df), len(test_df)


# In[24]:


train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[25]:


train_y = train_df.wins.values
val_y = val_df.wins.values
test_y = test_df.wins.values


# Feature Importance: Correlation Coefficient

# In[26]:


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

full_train.max()


# In[27]:


numerical = ['wins', 'losses', 'draws', 'height_cm', 'weight_in_kg', 'reach_in_cm', 'significant_strikes_landed_per_minute', 'significant_striking_accuracy', 'significant_strikes_absorbed_per_minute', 'significant_strike_defence', 'average_takedowns_landed_per_15_minutes', 'takedown_accuracy', 'takedown_defense', 'average_submissions_attempted_per_15_minutes']
full_train[numerical].corrwith(full_train.wins).abs()


# In[28]:


full_train[full_train.losses <= 2].wins.mean()


# In[29]:


full_train[(full_train.losses > 2) & (full_train.losses <= 12)].wins.mean()


# In[30]:


full_train[full_train.losses > 12].wins.mean()


# In[31]:


full_train[full_train.significant_strike_defence <= 20].wins.mean()


# In[32]:


full_train[(full_train.significant_strike_defence > 20) & (full_train.significant_strike_defence <= 50)].wins.mean()


# In[33]:


full_train[full_train.significant_strike_defence > 50].wins.mean()


# Word Cloud

# In[34]:


name_wins_dict = dict(zip(dfr['name'], dfr['wins']))

wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(name_wins_dict)

mp.figure(figsize=(10, 6))
mp.imshow(wordcloud, interpolation='bilinear')
mp.axis('off')
mp.title('Top Athletes Most Likely to Win')
mp.show()


# ## Train Models

# ### Logistic Regression Model

# In[35]:


import numpy as num
def sigmoid(z):
    return 1 / (1 + num.exp(-z))

z = num.linspace(-7, 7, 51)   

sigmoid(1000)


# In[36]:


import matplotlib.pyplot as mp
mp.plot(z, sigmoid(z))


# In[37]:


def log_reg(xi):
    score = w0

    for j in range(len(w)):
        result = result + xi[j] * w[j]

    return result


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[39]:


X = dfr[['losses', 
        'draws', 
        'height_cm', 
        'weight_in_kg', 
        'reach_in_cm', 
        'significant_strikes_landed_per_minute', 
        'significant_striking_accuracy', 
        'significant_strikes_absorbed_per_minute',                     
        'significant_strike_defence',                                 
        'average_takedowns_landed_per_15_minutes',                    
        'takedown_accuracy',                                          
        'takedown_defense',                                           
        'average_submissions_attempted_per_15_minutes']]

y = dfr['wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[40]:


import numpy as num

X_train = num.nan_to_num(X_train, nan=0)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)


# In[41]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg_model = LogisticRegression(random_state=42, max_iter=1000)  # You might need to adjust max_iter

log_reg_model.fit(X_train_scaled, y_train)

y_pred = log_reg_model.predict(X_test_scaled)


# Logisitc Regression Parameters

# In[45]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 500, 1000],
}


log_reg_model = LogisticRegression(random_state=42, max_iter=1000, solver='saga')

# Create Standard Scaler
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model
log_reg_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = log_reg_model.predict(X_test_scaled)


# ## Decision Tree Model

# In[46]:


from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=num.unique(y_train), y=y_train)


# In[47]:


import warnings

warnings.simplefilter(action='ignore', category=UserWarning)


# In[50]:


from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier 

class_weights = compute_class_weight('balanced', classes=num.unique(y_train), y=y_train)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

decision_tree_model = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(decision_tree_model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)


# Decision Tree Parameters

# In[51]:


best_params = grid_search.best_params_

final_model = DecisionTreeClassifier(random_state=42, **best_params)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

