import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMG_SIZE = (224, 224)
BATCH = 16

print("Loading dataset...")

dataset = tf.keras.utils.image_dataset_from_directory(
    "DFUC2021_test",
    image_size=IMG_SIZE,
    batch_size=BATCH
)

print("Building CNN from scratch...")

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(dataset.class_names), activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Starting training (scratch model)...")

model.fit(dataset, epochs=3)

print("Training complete.")



