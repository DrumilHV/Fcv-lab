import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('../resources/img_4.png')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to the grayscale image
edges = cv2.Canny(grayscale_image, threshold1=40, threshold2=200)

# Apply the Hough Line Transform to detect lines in the edge image
lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

# Draw detected lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the original image with detected lines
cv2.imshow('Original Image with Hough Lines', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
