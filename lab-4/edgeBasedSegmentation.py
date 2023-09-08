import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('../resources/img_6.png')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(grayscale_image, threshold1=30, threshold2=100)

# Find contours in the edge image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black image as a mask for the segmented objects
segmented_image = np.zeros_like(input_image)

# Draw contours on the segmented image
cv2.drawContours(segmented_image, contours, -1, (255, 192, 190), thickness=.5)

# Display the original and segmented images
cv2.imshow('Original Image', input_image)
cv2.imshow('Edge-Based Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
