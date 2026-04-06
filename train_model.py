import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("Algerian forest fires.csv")

df.columns = df.columns.str.strip()

df['Classes'] = df['Classes'].map({'fire':1, 'not fire':0})

df = df.dropna()

X = df[['day','month','year','Temperature','FWI','Classes',
        'RH','Ws','Rain','FFMC','DMC','ISI','DC','BUI']]

y = df['FWI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge()
model.fit(X_scaled, y)

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))

print("Model trained successfully")