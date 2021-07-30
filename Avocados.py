# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:35:21 2021

@author: johnp
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

avocado_data = pd.read_csv(('avocado.csv'), usecols = ["Date", "AveragePrice", "Total_Bags", "Total_Volume", "year"])

avocado_data

X_avocado = avocado_data.iloc[:, 0:5].values
X_avocado
X_avocado[0]

Y_avocado = avocado_data.iloc[:, 5].values
Y_avocado

#Label Encoder

label_encoder_labelavoc = LabelEncoder()

X_avocado[:,1]

labelavoc = label_encoder_labelavoc.fit_transform(X_avocado[:,1])

X_avocado[0]

label_encoder_Date = LabelEncoder()
label_encoder_AvaregePrice = LabelEncoder()
label_encoder_Total_Volume = LabelEncoder()
label_encoder_Total_Bags = LabelEncoder()
label_encoder_year = LabelEncoder()

X_avocado[:,2] = label_encoder_Date.fit_transform(X_avocado[:,2])
X_avocado[:,3] = label_encoder_AvaregePrice.fit_transform(X_avocado[:,3])
X_avocado[:,4] = label_encoder_Total_Volume.fit_transform(X_avocado[:,4])
X_avocado[:,8] = label_encoder_Total_Bags.fit_transform(X_avocado[:,8])
X_avocado[:,13] = label_encoder_year.fit_transform(X_avocado[:,13])

X_avocado[0]

X_avocado

#Standard Scaler

scaler_avocado = StandardScaler()
X_avocado = scaler_avocado.fit_transform(X_avocado)

X_avocado[0]

#Training and test

X_avocado_training, X_avocado_test, Y_avocado_training, Y_avocado_test = train_test_split(X_avocado, Y_avocado, test_size = 0.25, random_state = 0)

X_avocado_training.shape, Y_avocado_training.shape

X_avocado_test.shape, Y_avocado_test.shape

#Decision Tree Classifier
tree_avocado = DecisionTreeClassifier(criterion='entropy')
tree_avocado.fit(X_avocado, Y_avocado)

tree_avocado.feature_importances_

tree_avocado.classes_

prev = ["Date", "AveragePrice", "Total_Bags", "Total_Volume", "year"]
picture, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(tree_avocado, feature_names=prev, class_names = tree_avocado.classes_, filled=True);

prev = tree_avocado.predict(X_avocado_test)
prev