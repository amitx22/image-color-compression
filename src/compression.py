import numpy as np
from sklearn.cluster import KMeans
import joblib


def compress_image(pixels, n_colors=16, sample_size=10000):
    """
    Compress image using K-Means clustering.

    sample_size:
    Only a subset of pixels is used for training
    to make K-Means faster on large images.
    """

    # Sample pixels for training
    if len(pixels) > sample_size:
        indices = np.random.choice(
            len(pixels),
            sample_size,
            replace=False
        )

        training_pixels = pixels[indices]
    else:
        training_pixels = pixels

    # K-Means model
    kmeans = KMeans(
        n_clusters=n_colors,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    # Train
    kmeans.fit(training_pixels)

    # Assign every pixel to nearest cluster
    labels = kmeans.predict(pixels)

    # Replace pixels with cluster centers
    compressed_pixels = kmeans.cluster_centers_[labels]

    return compressed_pixels, kmeans


def save_model(model, path="models/kmeans_model.pkl"):
    """
    Save trained K-Means model.
    """

    joblib.dump(model, path)


def load_model(path="models/kmeans_model.pkl"):
    """
    Load saved K-Means model.
    """

    return joblib.load(path)


def calculate_elbow(pixels, k_values=None, sample_size=5000):
    """
    Calculate inertia for different K values.
    """

    if k_values is None:
        k_values = [2, 4, 8, 16, 32, 64]

    # Sample pixels
    if len(pixels) > sample_size:

        indices = np.random.choice(
            len(pixels),
            sample_size,
            replace=False
        )

        training_pixels = pixels[indices]

    else:
        training_pixels = pixels

    inertias = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=5,
            random_state=42
        )

        kmeans.fit(training_pixels)

        inertias.append(kmeans.inertia_)

    return k_values, inertias