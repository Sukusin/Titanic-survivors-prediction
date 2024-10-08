from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y_train = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100000, max_depth=5, random_state=1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission2.csv', index=False)

print('ready')

