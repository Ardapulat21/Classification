# Classification Project

## Overview

This project aims to compare the performance of three different classification algorithms: 
- Artificial Neural Networks (ANN)
- K-Nearest Neighbors (KNN)
- Decision Trees

The goal is to evaluate their effectiveness in classifying a given dataset and to analyze their respective test results.

## Dataset
The dataset contains 20,000 records.
<pre>
Customer ID          0
Age                  0
Gender               0
Loyalty Member       0
Product Type         0
SKU                  0
Rating               0
Order Status         0
Payment Method       0
Total Price          0
Unit Price           0
Quantity             0
Purchase Date        0
Shipping Type        0
Add-ons Purchased    0
Add-on Total         0

| Algorithm                           Accuracy (%) 
|----------------------------------|---------------|  
| Artificial Neural Networks (ANN) | 67%           |  
| K-Nearest Neighbors (KNN)        | 58%           |  
| Decision Tree                    | 55%           |
</pre>


## Data Preprocessing
<pre>
Unrelated attribues were dropped like 
'Customer ID',
'Add-ons Purchased',
'Add-on Total',
'Shipping Type',
'Quantity',
'Unit Price'
</pre>
by
<pre>
dataset.drop('Customer ID',axis=1)
dataset.drop('Add-ons Purchased',axis=1)
dataset.drop('Add-on Total',axis=1)
dataset.drop('Shipping Type',axis=1)
dataset.drop('Quantity',axis=1)
dataset.drop('Unit Price',axis=1)
</pre>

### Standardization
<pre>
scaler = StandardScaler()
X = scaler.fit_transform(dataset)
</pre>

### Label Encoding
<pre>
label_encoder = LabelEncoder()
categorical_columns = dataset.select_dtypes(include=['object']).columns

for col in categorical_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])
</pre>