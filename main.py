import tensorflow as tf
from tensorflow import keras
from sklearn import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pp

path = './Stars.csv'
ds = pd.read_csv(path)

# x, y = ds.drop(['Type', 'Color', 'Spectral_Class'], axis=1), ds['Type']
x = ds['A_M']
y = ds['Type']

x_train, x_test, y_train, y_test = x[:220],x[220:],y[:220],y[220:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(16, activation='leaky_relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
             )
              
model.fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test))

model.save_weights('weights.h5')
model.save('./model')

model.load_weights('weights.h5')

predictions = model.predict(x_test[:20])

# print predictions
print('preds: ')
print(np.argmax(predictions, axis=1))

# validate
print('actual data: ')
print(y_test[:20])
