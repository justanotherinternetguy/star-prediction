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
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
             )
              
history = model.fit(x_train, y_train, epochs=130, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print("SCORE")
print(score)

model.save_weights('weights')
model.save('./model')

model.load_weights('weights')

predictions = model.predict(x_test[:20])

# print predictions
print('preds: ')
print(np.argmax(predictions, axis=1))

# validate
print('actual data: ')
print(y_test[:20])
print(model.summary())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
