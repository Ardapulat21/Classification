import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()

dataset = pd.read_csv('Electronic_sales_Sep2023-Sep2024.csv');

dataset = dataset.drop('Customer ID',axis=1)
dataset = dataset.drop('Add-ons Purchased',axis=1)
dataset = dataset.drop('Add-on Total',axis=1)
dataset = dataset.drop('Shipping Type',axis=1)
dataset = dataset.drop('Quantity',axis=1)
dataset = dataset.drop('Unit Price',axis=1)

categorical_columns = dataset.select_dtypes(include=['object']).columns

for col in categorical_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])
    
Y = dataset['Order Status']

dataset = dataset.drop('Order Status',axis = 1)

scaler = StandardScaler()
X = scaler.fit_transform(dataset)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Decision Tree accuracy: {accuracy}")
