import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

def image_to_array(img):
    img = img.convert('RGB')
    img = np.array(img)
    img = img / 255.0
    pixels = img.reshape(-1, 3)
    return pixels

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class ManualKMeans:
    def __init__(self, n_clusters, max_iters=100, tolerance=1e-4):  # Perbaiki nama metode
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels_ = None
        
    def initialize_centroids(self, data):
        # Random initialization of centroids
        n_samples = data.shape[0]
        random_indices = random.sample(range(n_samples), self.n_clusters)
        self.centroids = data[random_indices]
        
    def assign_clusters(self, data):
        # Assign each point to nearest centroid
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((data - centroid) ** 2, axis=1))
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, data, labels):
        # Update centroids based on mean of assigned points
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def fit(self, data):
        # Initialize centroids
        self.initialize_centroids(data)
        
        for _ in range(self.max_iters):
            # Assign clusters
            old_centroids = np.copy(self.centroids)
            self.labels_ = self.assign_clusters(data)
            
            # Update centroids
            self.centroids = self.update_centroids(data, self.labels_)
            
            # Check for convergence
            if np.all(np.abs(old_centroids - self.centroids) < self.tolerance):
                break
                
        return self
    
    def predict(self, data):
        return self.assign_clusters(data)

def cluster_image(img, n_clusters):
    pixels = image_to_array(img)
    
    # Manual K-Means clustering
    kmeans = ManualKMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    
    # Get cluster centers and reshape pixels
    clustered_pixels = kmeans.centroids[kmeans.labels_]
    clustered_img = clustered_pixels.reshape(img.size[1], img.size[0], 3)
    clustered_img = (clustered_img * 255).astype(np.uint8)
    
    # Add labels to clusters
    labeled_img = Image.fromarray(clustered_img)
    draw = ImageDraw.Draw(labeled_img)

     try:
        font = ImageFont.truetype("arial.ttf", 30)  # Ukuran font 30px
    except IOError:
        font = ImageFont.load_default()  # Jika font tidak tersedia, gunakan default

    labels = kmeans.labels_.reshape(img.size[1], img.size[0])
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = np.argwhere(labels == label)
        if len(cluster_points) > 0:
            y, x = cluster_points[len(cluster_points) // 2]
            draw.text((x, y), str(label + 1), fill=(255, 0, 0), font=font,stroke_width=2, stroke_fill="white")
    
    return labeled_img, kmeans.labels_

# Streamlit app
st.title("Image Clustering using Manual K-Means")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Original Image', use_column_width=True)
    
    # Display clustered images for 2-5 clusters
    for n_clusters in range(2, 6):
        st.write(f"Clustering with {n_clusters} clusters")
        clustered_img, labels = cluster_image(img, n_clusters)
        st.image(clustered_img, caption=f'Clustered Image with {n_clusters} clusters', use_column_width=True)
    
    # Display pie chart for 5 clusters
    st.write("Cluster Visualization for 5 clusters")
    _, labels = cluster_image(img, 5)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.pie(counts, labels=unique_labels, autopct='%1.1f%%')
    st.pyplot(plt)
