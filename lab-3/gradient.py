import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('../resources/img_1.png', cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator to calculate gradients in x and y directions
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the gradient magnitude and direction
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
grad_direction = np.arctan2(grad_y, grad_x)

# Normalize the gradient magnitude to the range [0, 255]
grad_magnitude_normalized = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert the gradient direction to degrees
grad_direction_degrees = np.degrees(grad_direction)

# Display the gradient magnitude and direction
# cv2.imshow('Gradient Magnitude', grad_magnitude_normalized.astype(np.uint8))
cv2.imshow('Gradient Magnitude', grad_magnitude.astype(np.uint8))
cv2.imshow('Gradient Direction', grad_direction_degrees.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
