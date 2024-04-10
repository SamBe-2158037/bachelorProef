import numpy as np
import time
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras import layers, models
import baysopt as bo
from functools import partial
# Load your dataset or generate synthetic data
# For this example, let's assume you have your dataset loaded into X_train and y_train
PI = 3.141592
# Evaluate the model on test data

def build_model(config):
    model = models.Sequential()
    model.add(layers.Dense(config[1], activation='relu', input_shape=(1,)))
    for units in config[2:-1]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))  # Output layer
    return model

start_time = time.time()
conf = [1,16,16,1]


x=np.random.uniform(-1*PI,PI, 50)
trainingsdata = np.sin(x)+np.cos(x*5)
x2=np.random.uniform(-1*PI,PI, 50)
valii = np.sin(x2)+np.cos(x2*5)
netwerk =  build_model(conf)


def test(L):
    optimizerr = tf.keras.optimizers.Adam(learning_rate = L)
    netwerk.compile(optimizer=optimizerr,loss='mean_squared_error')
    netwerk.fit(x, trainingsdata, batch_size = 16, validation_split=0.5, verbose=0)
    y_pred = netwerk.predict(x)
    R2 = r2_score(valii,y_pred)
    print(R2)
    return R2

bo.search(np.arange(0.01,0.5,0.01),test) 


print("--- %s seconds ----- end at: "% (time.time() - start_time), end="")
print(time.asctime)
