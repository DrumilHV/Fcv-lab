import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('../resources/img_3.png')

# Convert the image to the HSV color space (Hue, Saturation, Value)
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the color range you want to segment (in HSV)
lower_color = np.array([30, 50, 50])  # Example: lower bound for green
upper_color = np.array([90, 255, 255])  # Example: upper bound for green

# Create a mask to identify pixels within the specified color range
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Apply the mask to the original image to extract the segmented region
segmented_image = cv2.bitwise_and(input_image, input_image, mask=color_mask)

# Display the original and segmented images
cv2.imshow('Original Image', input_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
