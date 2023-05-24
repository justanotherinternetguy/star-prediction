import tensorflow as tf
from tensorflow import keras
from sklearn import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pp

dict = {0:"red dwarf", 1:"brown dwarf", 2:"white dwarf", 3:"main seq", 4:"supergiant", 5:"hypergiant"}

path = './Stars.csv'
ds = pd.read_csv(path)

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
     )
    return model


model = create_model()

model.load_weights('weights')

temperature = float(input("temp (k) >> "))
AM = float(input("AM >> "))
A_M = np.array([[AM]])
predictions = model.predict(A_M)

print('preds: ')
print(np.argmax(predictions, axis=1))
for i in np.argmax(predictions, axis=1):
    print(dict[i])
