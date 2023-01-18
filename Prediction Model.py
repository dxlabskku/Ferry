
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


# # Load data

# In[3]:


ferry = pd.read_excel('/home/ubuntu/jupyter/seongkyu/data/ferry.xlsx')
weather = pd.read_excel('/home/ubuntu/jupyter/seongkyu/data/weather.xlsx')


# # Preprocessing : Missing value, Min-Max

# In[4]:


# imputing missing value with knn_imputer
imputer = KNNImputer(n_neighbors=10)
imputed = imputer.fit_transform(weather.iloc[:,2:])
weather = pd.concat([weather.iloc[:,:2], pd.DataFrame(imputed)], axis=1)

# replacing with Min-Max scaler
scaler_ = MinMaxScaler()
scaled = scaler_.fit_transform(weather.iloc[:,2:])
weather = pd.concat([weather.iloc[:,:2], pd.DataFrame(scaled)], axis=1)


# In[5]:


#replace col
weather_cols = ['Point', 'Time', 'Wind speed (m/s)', 'Wind direction (deg)', 'GUST wind speed (m/s)', 'Local pressure (hPa)', 'Humidity (%)',
                'Temperature (°C)', 'Water temperature (°C)', 'Maximum wave height (m)', 'Significant wave height (m)', 'Average wave height (m)', 'Wave period (sec)',
                'Wave Direction (deg)']

ferry_cols = ['Time','Company','Ferry','Target']


weather.columns = weather_cols
ferry.columns = ferry_cols


# # The datasets(ferry and Meteorological) are paired

# In[27]:


col_num = 0
data_list = []


# Sequentially applying meteorological datasets from H hours before ferry departur
H = 48   


for i in tqdm(range(len(ferry))):
    try:
        y = str(datetime.datetime.strptime(ferry['Time'][i], '%Y-%m-%d %H:%M') - datetime.timedelta(hours=H))[:4]
        m = str(datetime.datetime.strptime(ferry['Time'][i], '%Y-%m-%d %H:%M') - datetime.timedelta(hours=H))[5:7]
        d = str(datetime.datetime.strptime(ferry['Time'][i], '%Y-%m-%d %H:%M') - datetime.timedelta(hours=H))[8:10]
        h = str(datetime.datetime.strptime(ferry['Time'][i], '%Y-%m-%d %H:%M') - datetime.timedelta(hours=H))[11:16]
        W = weather.loc[weather['Time'] == str(y)+'-'+str(m)+'-'+str(d)+' '+str(h)].values.tolist()
        T = 1 if ferry['Target'][i] == '정상' else 0
        data_list.extend(W[0])
        data_list.append(ferry['Company'][i])
        data_list.append(T)
        col_num = col_num + 1
    except:
        pass

Data = pd.DataFrame(np.array(data_list).reshape(col_num, 16))
Data.columns = ['Point', 'Time', 'Wind speed (m/s)', 'Wind direction (deg)', 'GUST wind speed (m/s)', 'Local pressure (hPa)', 'Humidity (%)',
               'Temperature (°C)', 'Water temperature (°C)', 'Maximum wave height (m)', 'Significant wave height (m)', 'Average wave height (m)', 'Wave period (sec)',
               'Wave Direction (deg)', 'Company', 'Target']


# # Train_Test Split

# In[34]:


x_data = Data.iloc[:,2:14]
y_data = Data.iloc[:,15:16]

# data split (train-test)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=10, stratify = y_data)

print('Test data : {0}'.format(len(x_test)))
print('Train data : {0}'.format(len(x_train)))


# # Sampling

# #### Over-sampling

# In[ ]:


over = SMOTE(random_state=10)
x_train, y_train = over.fit_resample(x_train,y_train)
print('[Over-Sampling]Train data : {0}'.format(len(x_train)))


# ## OR

# #### Under-sampling

# In[ ]:


under = NearMiss()
x_train, y_train = under.fit_resample(x_train,y_train)
print('[Under-Sampling]Train data : {0}'.format(len(x_train)))


# # Model evaluation

# In[35]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ### Parameter Optimization

# In[37]:


from sklearn.model_selection import GridSearchCV

# classifiers
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
KNN = KNeighborsClassifier()
Ada = AdaBoostClassifier()


RF_param = {'n_estimators':[100,300,500],
            'criterion':['entropy','log_loss','gini'],
            'min_samples_split':[2,5,10]}

DT_param = {'criterion':['entropy','log_loss','gini'],
            'splitter':['best','random'],
            'min_samples_split':[2,5,10]}

KNN_param = {'n_neighbors':[5,10,15],
             'weights':['uniform','distance'],
             'leaf_size':[30,50,70]}

Ada_param = {'n_estimators':[50,100,150],
             'learning_rate':[1, 1.5, 2]}


DT_grid_param = GridSearchCV(DT, DT_param, cv=5)
DT_grid_param.fit(x_train,y_train)

RF_grid_param = GridSearchCV(RF, RF_param, cv=5)
RF_grid_param.fit(x_train,y_train)

KNN_grid_param = GridSearchCV(KNN, KNN_param, cv=5)
KNN_grid_param.fit(x_train,y_train)

Ada_grid_param = GridSearchCV(Ada, Ada_param, cv=5)
Ada_grid_param.fit(x_train,y_train)


# In[38]:


# Best parameters
print(DT_grid_param.best_params_)
print(RF_grid_param.best_params_)
print(KNN_grid_param.best_params_)
print(Ada_grid_param.best_params_)


# ### 5-Fold CV

# In[43]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(x_train)


# In[55]:


# Decision Tree
PO_DT = DecisionTreeClassifier(criterion='gini', min_samples_split=2, splitter='best')

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    PO_DT.fit(x_train.iloc[train_index], y_train.iloc[train_index])

print(classification_report(y_test, PO_DT.predict(x_test), target_names = ['0','1'], digits=4))


# In[48]:


#Random Forest
PO_RF = RF = RandomForestClassifier(criterion='gini', min_samples_split=5, n_estimators=100)

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    PO_RF.fit(x_train.iloc[train_index], y_train.iloc[train_index])

print(classification_report(y_test, PO_RF.predict(x_test), target_names = ['0','1'], digits=4))


# In[49]:


#KNN
PO_KNN = KNeighborsClassifier(leaf_size=30, n_neighbors=5, weights='distance')

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    PO_KNN.fit(x_train.iloc[train_index], y_train.iloc[train_index])

print(classification_report(y_test, PO_KNN.predict(x_test), target_names = ['0','1'], digits=4))


# In[50]:


# AdaBoost
PO_Ada = AdaBoostClassifier(learning_rate=1.5, n_estimators=150)

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    PO_Ada.fit(x_train.iloc[train_index], y_train.iloc[train_index])

print(classification_report(y_test, PO_Ada.predict(x_test), target_names = ['0','1'], digits=4))

