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

x = ds['A_M']
y = ds['Type']

x_train, x_test, y_train, y_test = x[:220],x[220:],y[:220],y[220:]

model = create_model()
model = model.load_weights('weights.h5')

predictions = model.predict(x_test[:5])

# print predictions
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# validate
print(y_test[:5]) # [7, 2, 1, 0, 4]

