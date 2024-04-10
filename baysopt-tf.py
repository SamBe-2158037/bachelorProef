import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

from sklearn.metrics import r2_score
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras import layers, models

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

PI = 3.141592

def build_model(config):
    model = models.Sequential()
    model.add(layers.Dense(config[1], activation='relu', input_shape=(1,)))
    for units in config[2:-1]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))  # Output layer
    return model

def test(L, netwerk, trainingsdata, valii):
    optimizer = tf.keras.optimizers.Adam(learning_rate=L)  # Corrected lr parameter
    netwerk.compile(optimizer=optimizer, loss='mean_squared_error')
    netwerk.fit(x, trainingsdata, batch_size=16, validation_split=0.5, verbose=0, epochs=10)
    y_pred = netwerk.predict(x)
    R2 = r2_score(valii, y_pred)
    return R2

def plot_bo(bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()

start_time = time.time()
conf = [1, 16, 16, 1]

x = np.random.uniform(-1 * PI, PI, 50)
trainingsdata = np.sin(x) + np.cos(x * 5)
x2 = np.random.uniform(-1 * PI, PI, 50)
valii = np.sin(x2) + np.cos(x2 * 5)
netwerk = build_model(conf)

p = {'L': (0.1, 0.5)}
bo = BayesianOptimization(test, pbounds=p, verbose=7)
bo.maximize(init_points=5, n_iter=10, acq='ucb', kappa=0.1)

plot_bo(bo)

print("--- %s seconds ----- end at: " % (time.time() - start_time), end="")
print(time.asctime)
