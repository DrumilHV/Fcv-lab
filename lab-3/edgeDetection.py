import cv2

# Load the input image
input_image = cv2.imread('../resources/img_6.png')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(grayscale_image, threshold1=30, threshold2=80)

# Display the original and edge-detected images
cv2.imshow('Original Image', input_image)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
