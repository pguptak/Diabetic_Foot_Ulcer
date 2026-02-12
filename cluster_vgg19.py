import os
import numpy as np

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATASET_PATH = "DFUC2021_test"

print("\nLoading VGG19 model...")

model = VGG19(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("\nMODEL SUMMARY:")
model.summary()

print("\nExtracting features from images...")

features = []
filenames = []

files = os.listdir(DATASET_PATH)

for i, file in enumerate(files):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):

        path = os.path.join(DATASET_PATH, file)

        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        feat = model.predict(arr, verbose=0)

        features.append(feat.flatten())
        filenames.append(file)

        if i % 200 == 0:
            print("Processed:", i)

features = np.array(features)

print("Feature matrix shape:", features.shape)

print("\nRunning PCA...")

pca = PCA(n_components=100)
reduced = pca.fit_transform(features)

print("Reduced feature shape:", reduced.shape)

print("\nRunning KMeans clustering...")

k = 5   # you can change cluster count later
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(reduced)

print("Clustering complete.")

clusters = {}

for file, label in zip(filenames, labels):
    clusters.setdefault(label, []).append(file)

for c in clusters:
    print(f"\nCluster {c} — {len(clusters[c])} images")






