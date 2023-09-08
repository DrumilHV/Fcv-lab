import cv2
import numpy as np

# Load and preprocess the image
image = cv2.imread('../resources/img_3.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image
pixels = image.reshape((-1, 3))

# Perform K-means clustering
K = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels.astype(np.float32), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Reshape the labels
segmented_image = labels.reshape(image.shape[:2])
# segmented_image = segmented_image.astype(np.unit8)
# cv2.imshow('segmented_image',segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create a colored segmented image
cluster_colors = [
    [0, 0, 255],   # Blue
    [255, 0, 0],   # Red
    [0, 255, 0],   # Green
]
segmented_image_colored = np.zeros_like(image)
for i in range(K):
    mask = (segmented_image == i)
    # color = np.random.randint(0, 256, 3)  # Random color for each cluster
    color = cluster_colors[i]
    segmented_image_colored[mask] = color

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
