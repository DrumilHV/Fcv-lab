import cv2
import numpy as np

# Load the input image
input_img = cv2.imread('../resources/specified_image.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 3)

# Calculate the unsharp mask by subtracting the blurred image from the original grayscale image
unsharp_mask = cv2.addWeighted(gray_img, 2.5, blurred_img, -1.5, 0)

# Convert the unsharp mask back to a color image
output_img = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)

# Display the input and output images
cv2.imshow('Input Image', input_img)
cv2.imshow('Unsharp Masking', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
