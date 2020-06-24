# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 23:17:49 2020

@author: User
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',100)

df_train=pd.read_csv('train values.csv')
df_test=pd.read_csv('test values.csv')
df_labels=pd.read_csv('train labels.csv')

#DEALING WITH TRAIN CATEGORY
#convert date column to date category
df_train.date_recorded=pd.to_datetime(df_train.date_recorded)

#to check for duplicated rows
print(df_train.index.duplicated().sum())
#set index to column id
df_train.set_index('id',inplace=True)
#select object datatype
df_train_str=df_train.select_dtypes(include=['object'])

#represent the data value  for str on a bar chart
#for i in df_train_str.columns:
#    df_train_str[i].value_counts().plot(kind='bar')
#    plt.show()
    
#Missing and Zero values for integer columns
print(df_train.isnull().sum())#for missing values
missing_values=['funder','installer','subvillage','public_meeting','scheme_management','scheme_name','permit']
for i in missing_values:
    df_train[i]=df_train[i].fillna(df_train[i].value_counts().index[0])

#for values with 0 or 1
mean_list=['amount_tsh','longitude', 'latitude', 'num_private']
for i in mean_list:
    df_train[i].loc[df_train[i]==0]=df_train[i].mean()

hi_list=['gps_height','region_code', 'district_code','construction_year',]
for i in hi_list:
    if df_train[i].value_counts().index[0] != 0:
        df_train[i].loc[df_train[i]==0]=df_train[i].value_counts().index[0]
    else:
        df_train[i].loc[df_train[i]==0]=df_train[i].value_counts().index[1]

#create lifespan column
df_train['lifespan']=df_train.date_recorded.dt.year - df_train.construction_year
#drop the date column
df_train.drop('date_recorded',axis=1,inplace=True)
#for population <5
df_train['population'].loc[df_train['population']<=5]=df_train[i].mean()    
#scale data
scale_cols=['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private','region_code', 'district_code', 'population', 'construction_year','lifespan']
for i in scale_cols:
    df_train[i]=(df_train[i]-df_train[i].min())/(df_train[i].max()-df_train[i].min())    

#carry out binary encodeing for data
import category_encoders as ce
cols=['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region','lga', 'ward', 'public_meeting', 'recorded_by', 'scheme_management','scheme_name', 'permit', 'extraction_type', 'extraction_type_group','extraction_type_class', 'management', 'management_group', 'payment','payment_type', 'water_quality', 'quality_group', 'quantity','quantity_group', 'source', 'source_type', 'source_class','waterpoint_type', 'waterpoint_type_group']
encoder=ce.BinaryEncoder(cols=cols)
df_train=encoder.fit_transform(df_train)

#work on label file
df_labels.set_index('id',inplace=True)
encoder=ce.BinaryEncoder(cols=df_labels)
df_labels=encoder.fit_transform(df_labels)

from sklearn.model_selection import train_test_split
Xtrain,Xval,Ytrain,Yval=train_test_split(df_train,df_labels,test_size=0.3,random_state=0)

from sklearn.linear_model import Ridge
model=Ridge(alpha=0.1)
model.fit(Xtrain,Ytrain)
pred=model.predict(Xval)


#DEALING WITH TEST CATEGORY
#convert date column to date category
df_test.date_recorded=pd.to_datetime(df_test.date_recorded)

#to check for duplicated rows
print(df_test.index.duplicated().sum())
#set index to column id
df_test.set_index('id',inplace=True)

    
#Missing and Zero values for integer columns
print(df_test.isnull().sum())#for missing values
missing_values=['funder','installer','subvillage','public_meeting','scheme_management','scheme_name','permit']
for i in missing_values:
    df_test[i]=df_test[i].fillna(df_test[i].value_counts().index[0])

#for values with 0 or 1
mean_list=['amount_tsh','longitude', 'latitude', 'num_private']
for i in mean_list:
    df_test[i].loc[df_test[i]==0]=df_test[i].mean()

hi_list=['gps_height','region_code', 'district_code','construction_year',]
for i in hi_list:
    if df_test[i].value_counts().index[0] != 0:
        df_test[i].loc[df_test[i]==0]=df_test[i].value_counts().index[0]
    else:
        df_test[i].loc[df_test[i]==0]=df_test[i].value_counts().index[1]

#create lifespan column
df_test['lifespan']=df_test.date_recorded.dt.year - df_test.construction_year
#drop the date column
df_test.drop('date_recorded',axis=1,inplace=True)
#for population <5
df_test['population'].loc[df_test['population']<=5]=df_test[i].mean()    
#scale data
scale_cols=['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private','region_code', 'district_code', 'population', 'construction_year','lifespan']
for i in scale_cols:
    df_test[i]=(df_test[i]-df_test[i].min())/(df_test[i].max()-df_test[i].min())    

#carry out binary encodeing for data
import category_encoders as ce
cols=['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region','lga', 'ward', 'public_meeting', 'recorded_by', 'scheme_management','scheme_name', 'permit', 'extraction_type', 'extraction_type_group','extraction_type_class', 'management', 'management_group', 'payment','payment_type', 'water_quality', 'quality_group', 'quantity','quantity_group', 'source', 'source_type', 'source_class','waterpoint_type', 'waterpoint_type_group']
encoder=ce.BinaryEncoder(cols=cols)
df_test=encoder.fit_transform(df_test)

#to predict the test data using training model
model.predict(df_test)