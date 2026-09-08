import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_dominant_colors(image_path, k=5):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    
    pixels = img.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def main():
    print("Starting Color Palette Extractor...")
    
    data_dir = "./data/images"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print("Please add a test image (.jpg or .png) to this folder and run again.")
        return

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(data_dir, filename)
            palette = extract_dominant_colors(filepath)
            
            if palette is not None:
                print("-" * 30)
                print(f"Dominant Palette for {filename}:")
                for i, color in enumerate(palette):
                    print(f"Color {i+1} RGB: {tuple(color)}")

if __name__ == "__main__":
    main()
