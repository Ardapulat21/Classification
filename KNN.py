#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 03:13:57 2025

@author: arda
"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)


y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Model accuracy: {accuracy}")


