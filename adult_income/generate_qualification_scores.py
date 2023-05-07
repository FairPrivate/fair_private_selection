import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

df = pd.read_csv('adult.csv') #

#df = df[df.loc[(df['race'] == 'White') and (df['race'] == 'Black')]]
#df = df[df['race'] == ['White', 'Black']]
df = df[(df != '?').all(axis=1)]
array = ['White', 'Black']
df = df.loc[df['race'].isin(array)]

# Get one hot encoding of columns B
one_hot = pd.get_dummies(df['workclass'])
# Drop column B as it is now encoded
df = df.drop('workclass', axis=1)
# Join the encoded df
df = df.join(one_hot)

one_hot = pd.get_dummies(df['education'])
df = df.drop('education',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['marital-status'])
df = df.drop('marital-status',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['occupation'])
df = df.drop('occupation',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['relationship'])
df = df.drop('relationship',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['native-country'])
df = df.drop('native-country',axis = 1)
df = df.join(one_hot)
label_income = df['income']

race = pd.Series(np.where(df.race.values == 'White', 1, 0),
          df.index)
df = df.drop('race', axis = 1)
df['race'] = race

gender = pd.Series(np.where(df.gender.values == 'Male', 1, 0),
          df.index)
df = df.drop('gender', axis = 1)
df['gender'] = gender

label = pd.Series(np.where(df.income.values == '>50K', 1, 0),
          df.index)
df['label'] = label
df = df.drop('income', axis = 1)  # 43131
df_w = df[df['race'] == 'White']  # 38903
df_b = df[df['race'] == 'Black']  # 4228
print ('hi')

df = df.drop('label', axis = 1)
# Split our data
train, test, train_labels, test_labels = train_test_split(df,
                                                          label,
                                                          test_size=0.001,
                                                          random_state=1)

print ('hi')
'''
rf = xgb.XGBClassifier()
rf.fit(df, label)
a = rf.predict_proba(df)


filename = 'xgb_model.pkl'
pickle.dump(rf, open(filename, 'wb'))
'''
filename = 'xgb_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(df, label)
print ('hi')

'''
df['score'] = a[:, 1]
df['label'] = label
df.to_csv('df.csv')
'''



