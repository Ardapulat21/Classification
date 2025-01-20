#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 03:13:57 2025

@author: arda
"""
import pandas as pd

pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import tensorflow as tf


label_encoder = LabelEncoder()

dataset = pd.read_csv('Electronic_sales_Sep2023-Sep2024.csv')

dataset = dataset.drop('Customer ID',axis=1)
dataset = dataset.drop('Add-ons Purchased',axis=1)
dataset = dataset.drop('Add-on Total',axis=1)
dataset = dataset.drop('Shipping Type',axis=1)
dataset = dataset.drop('Quantity',axis=1)
dataset = dataset.drop('Unit Price',axis=1)

categorical_columns = dataset.select_dtypes(include=['object']).columns

for col in categorical_columns:
    dataset[col] = to_categorical(label_encoder.fit_transform(dataset[col]))
    

Y = dataset['Order Status']
dataset = dataset.drop('Order Status',axis = 1)


scaler = StandardScaler()
X = scaler.fit_transform(dataset)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Fitting ANN
ann.fit(X_train,y_train,batch_size=32,epochs = 100)