import cv2
import numpy as np

def harris_corner_detector(image, k=0.04, threshold=0.01):
    # Step 1: Compute image gradients
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Compute the elements of the structure tensor
    Ixx = gradient_x ** 2
    Ixy = gradient_x * gradient_y
    Iyy = gradient_y ** 2

    # Step 3: Compute the corner response function
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    corner_response = det_M - k * (trace_M ** 2)

    # Step 4: Find corner points above the threshold
    corners = np.where(corner_response > threshold)

    # Step 5: Create an output image with detected corners
    output_image = np.copy(image)
    output_image[corners] = [0, 0, 255]  # Mark corners in red

    return output_image

# Load the input image
input_image = cv2.imread('input_image.jpg')

# Detect Harris corners
output_image = harris_corner_detector(input_image)

# Display the original and Harris corner-detected images
cv2.imshow('Original Image', input_image)
cv2.imshow('Harris Corner Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
