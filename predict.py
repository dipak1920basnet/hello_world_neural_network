import tensorflow as tf 
import numpy as np 

loaded_model = tf.keras.models.load_model("my_model.keras")

predictions = loaded_model.predict(np.array([10.0]))
print(predictions)