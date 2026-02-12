import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMG_SIZE = (128,128)
BATCH = 16

print("Loading images...")

dataset = tf.keras.utils.image_dataset_from_directory(
    "DFUC2021_test",
    labels=None,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

dataset = dataset.map(lambda x: (x/255.0, x/255.0))

print("Building autoencoder from scratch...")

encoder = models.Sequential([
    layers.Conv2D(32,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu",padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation="relu",padding="same"),
    layers.MaxPooling2D()
])

decoder = models.Sequential([
    layers.Conv2DTranspose(128,3,strides=2,activation="relu",padding="same"),
    layers.Conv2DTranspose(64,3,strides=2,activation="relu",padding="same"),
    layers.Conv2DTranspose(32,3,strides=2,activation="relu",padding="same"),
    layers.Conv2D(3,3,activation="sigmoid",padding="same")
])

inputs = layers.Input(shape=(128,128,3))
encoded = encoder(inputs)
decoded = decoder(encoded)

autoencoder = models.Model(inputs, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

print("Training autoencoder from scratch...")

autoencoder.fit(dataset, epochs=5)

print("Training complete.")

print("Saving model...")

autoencoder.save("models/autoencoder_v1.keras")

print("Model saved.")



