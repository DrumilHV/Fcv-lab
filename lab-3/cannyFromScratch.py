import numpy as np
import cv2

# Load the input image in grayscale
input_image = cv2.imread('../resources/img_4.png', cv2.IMREAD_GRAYSCALE)

# Define Gaussian smoothing kernel
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])

# Convolve the input image with the Gaussian kernel
smoothed_image = cv2.filter2D(input_image, -1, gaussian_kernel)

# Compute the gradient magnitude and direction using Sobel filters
gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# Non-maximum suppression
height, width = gradient_magnitude.shape
edge_image = np.zeros_like(gradient_magnitude)

for i in range(1, height - 1):
    for j in range(1, width - 1):
        angle = gradient_direction[i, j]
        if (angle >= -np.pi/8 and angle <= np.pi/8) or (angle >= 7*np.pi/8 or angle <= -7*np.pi/8):
            if (gradient_magnitude[i, j] >= gradient_magnitude[i, j+1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i, j-1]):
                edge_image[i, j] = gradient_magnitude[i, j]
        elif (angle >= np.pi/8 and angle <= 3*np.pi/8) or (angle >= -7*np.pi/8 and angle <= -5*np.pi/8):
            if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j+1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j-1]):
                edge_image[i, j] = gradient_magnitude[i, j]
        elif (angle >= 3*np.pi/8 and angle <= 5*np.pi/8) or (angle >= -5*np.pi/8 and angle <= -3*np.pi/8):
            if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j]):
                edge_image[i, j] = gradient_magnitude[i, j]
        else:
            if (gradient_magnitude[i, j] >= gradient_magnitude[i-1, j-1]) and (gradient_magnitude[i, j] >= gradient_magnitude[i+1, j+1]):
                edge_image[i, j] = gradient_magnitude[i, j]

# Double thresholding and edge tracking by hysteresis
low_threshold = 30
high_threshold = 100

strong_edges = (edge_image > high_threshold)
weak_edges = (edge_image >= low_threshold) & (edge_image <= high_threshold)

edge_image_final = np.zeros_like(edge_image)
edge_image_final[strong_edges] = 255

# Display the final edge-detected image
cv2.imshow('Canny Edge Detection', edge_image_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
