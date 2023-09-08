import cv2
import numpy as np

# Load the image
image = cv2.imread('../resources/img_4.png', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display the original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the equalized image
cv2.imwrite('equalized_image.jpg', equalized_image)
