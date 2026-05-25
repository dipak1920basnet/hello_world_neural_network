import tensorflow as tf 
import numpy as np 

model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(1,)), #Decide the shape of input 
        tf.keras.layers.Dense(units=1)
    ]
)

model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([    -7, -6, -5, -4, -3,
    -2, -1, 0, 1, 2,
    3, 4, 5, 6, 7], dtype=float)
ys = np.array([-13, -11, -9, -7, -5,
    -3, -1, 1, 3, 5,
    7, 9, 11, 13, 15], dtype=float)

model.fit(xs,ys, epochs=500)

# Save the model with name my_model
model.save("my_model.keras")


# predictions = model.predict([10.0])

predictions = model.predict(np.array([10.0]))
print(predictions)