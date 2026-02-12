import os
import numpy as np


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture



DATASET_PATH = "DFUC2021_test"

print("\nLoading MobileNetV3 model...")

model = MobileNetV3Large(
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

print("\nRunning Agglomerative (Ward) clustering...")

agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
agg_labels = agg.fit_predict(reduced)

agg_clusters = {}

for file, label in zip(filenames, agg_labels):
    agg_clusters.setdefault(label, []).append(file)

for c in agg_clusters:
    print(f"\nAgg Cluster {c} — {len(agg_clusters[c])} images")

print("\nRunning DBSCAN clustering...")

db = DBSCAN(eps=8.0, min_samples=5)
db_labels = db.fit_predict(reduced)

db_clusters = {}
noise_count = 0

for file, label in zip(filenames, db_labels):
    if label == -1:
        noise_count += 1
    else:
        db_clusters.setdefault(label, []).append(file)

for c in db_clusters:
    print(f"\nDBSCAN Cluster {c} — {len(db_clusters[c])} images")

print("\nDBSCAN noise points —", noise_count)

print("\nRunning Gaussian Mixture Model clustering...")

gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(reduced)

gmm_clusters = {}

for file, label in zip(filenames, gmm_labels):
    gmm_clusters.setdefault(label, []).append(file)

for c in gmm_clusters:
    print(f"\nGMM Cluster {c} — {len(gmm_clusters[c])} images")






