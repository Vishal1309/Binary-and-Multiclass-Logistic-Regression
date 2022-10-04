from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid')
])
