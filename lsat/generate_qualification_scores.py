import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score

df = pd.read_csv('law_data.csv')

array = ['White', 'Black']
df = df.loc[df['race'].isin(array)]

df = df.drop('Unnamed: 0', axis=1)
df = df. drop('region_first', axis=1)
df = df. drop('sander_index', axis=1)
#df = df. drop('first_pf', axis=1)

race = pd.Series(np.where(df.race.values == 'White', 1, 0), df.index)
df = df.drop('race', axis = 1)
df['race'] = race

sex = pd.Series(np.where(df.sex.values == 1, 1, 0), df.index)
df = df.drop('sex', axis = 1)
df['sex'] = sex

label = df['first_pf']
df['label'] = label
df = df. drop('first_pf', axis=1)

df.to_csv('lsat_df_csv.csv')
df_w = df[df['race'] == 1]  # 18285
df_b = df[df['race'] == 0]  # 1282
print ('hi')

df = df.drop('label', axis = 1)
# Split our data
train, test, train_labels, test_labels = train_test_split(df,
                                                          label,
                                                          test_size=0.1,
                                                          random_state=4)

print ('hi')

rf = xgb.XGBClassifier()
rf.fit(df, label)
a = rf.predict_proba(df)


#print("Accuracy: ", accuracy_score(label, a))
filename = 'xgb_model_lsat.pkl'
pickle.dump(rf, open(filename, 'wb'))

'''
filename = 'xgb_model_lsat.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(df, label)  # 0.9274
print ('hi')
'''

df['score'] = a[:, 1]
df['label'] = label
df.to_csv('lsat_df.csv')
print ('hi')




