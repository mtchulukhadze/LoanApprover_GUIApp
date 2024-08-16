
# svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

path = r"D:\Data\AI & ML\train_u6lujuX_CVtuZ9i.csv"
data = pd.read_csv(path)
print(data.head())

data = data.dropna()
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})
data['Dependents'] = data['Dependents'].replace({"3+": 4})
data['Married'] = data['Married'].replace({"Yes": 1, "No": 0})
data['Gender'] = data['Gender'].replace({"Male": 1, "Female": 0})
data['Self_Employed'] = data['Self_Employed'].replace({"Yes": 1, "No": 0})
data['Property_Area'] = data['Property_Area'].replace({"Rural": 0, "Semiurban": 1, "Urban": 2})
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})

print(data['Dependents'].value_counts())

sns.countplot(x='Education', hue='Loan_Status', data=data)

y = data['Loan_Status']
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
data_accuracy = accuracy_score(y_test, y_predict)
print(data_accuracy)



# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()

path = r"D:\Data\AI & ML\train_u6lujuX_CVtuZ9i.csv"
data = pd.read_csv(path)


data = data.dropna()
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})
data['Dependents'] = data['Dependents'].replace({"3+": 4})
data['Married'] = data['Married'].replace({"Yes": 1, "No": 0})
data['Gender'] = data['Gender'].replace({"Male": 1, "Female": 0})
data['Self_Employed'] = data['Self_Employed'].replace({"Yes": 1, "No": 0})
data['Property_Area'] = data['Property_Area'].replace({"Rural": 0, "Semiurban": 1, "Urban": 2})
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})


y = data['Loan_Status']
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(units=100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=50, activation='relu'))
# model.add(Dropout(rate=0.3))
model.add(Dense(units=40, activation='relu'))
model.add(GaussianNoise(stddev=0.99))
model.add(Dense(units=30, activation='relu'))

# how to add layer, that have one output, binar
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=30, epochs = 100, validation_data=(X_test, y_test), verbose=1)

model.save('loan_approval_model.h5')


import joblib

joblib.dump(scaler, 'scaler.pkl')

# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

path = r"D:\Data\AI & ML\train_u6lujuX_CVtuZ9i.csv"
data = pd.read_csv(path)
print(data.head())

data = data.dropna()
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})
data['Dependents'] = data['Dependents'].replace({"3+": 4})
data['Married'] = data['Married'].replace({"Yes": 1, "No": 0})
data['Gender'] = data['Gender'].replace({"Male": 1, "Female": 0})
data['Self_Employed'] = data['Self_Employed'].replace({"Yes": 1, "No": 0})
data['Property_Area'] = data['Property_Area'].replace({"Rural": 0, "Semiurban": 1, "Urban": 2})
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})


y = data['Loan_Status']
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
