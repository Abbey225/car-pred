import numpy as np
import pandas as pd

df=pd.read_csv('carprice_encoded.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())

x=df.drop('selling_price',axis=1)
y=df.selling_price

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor()
rf_model.fit(x_train,y_train)
prediction = rf_model.predict(x_test)

import pickle

pickle.dump(rf_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))