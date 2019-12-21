#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

train_data = pd.read_csv("C:/Users/DELL/Downloads/boston_train.csv")

test_data = pd.read_csv("C:/Users/DELL/Downloads/boston_test.csv")

predict_data = pd.read_csv("C:/Users/DELL/Downloads/boston_predict.csv")

train_data.head()

FEATUREES=['CRIM','ZN','INDUS', 'NOX','RM', 'AGE','DIS','TAX','PTRATIO','MEDV']
LABEL="MEDV"

plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True)

X_train = train_data[['CRIM','ZN','INDUS', 'NOX','RM','AGE','TAX','PTRATIO']]
Y_train = train_data['MEDV']

# ## Preprocessing

from sklearn.preprocessing import normalize, scale
X_scale = scale(X_train)
X_norm = normalize(X_scale)

X_scale

# ## BUILD AND TRAIN

def build_model():
    model = keras.Sequential([
        layers.Dense(64,activation=tf.nn.relu,
                    input_shape=[len(X_train.keys())]),
        layers.Dense(32,activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
             optimizer=optimizer,
             metrics=['mean_absolute_error',
                     'mean_squared_error'])
    return model

model= build_model()
model.summary()

EPOCHS=1000
history = model.fit(X_norm,Y_train,
                   epochs=EPOCHS,validation_split=0.2,
                    verbose=0
                   )


# ## TESTING THE MODEL

X_test = test_data[['CRIM','ZN','INDUS', 'NOX','RM','AGE','TAX','PTRATIO']]


Y_test = test_data['MEDV']

X_scale_test = scale(X_test)
X_norm_test = normalize(X_scale_test)
yhat = model.predict(X_norm_test)

from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(Y_test,yhat)
print(MSE)
r2_score(Y_test,yhat)

