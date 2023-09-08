import cv2
import numpy as np

# Load the input image and reference image
input_image = cv2.imread('../resources/img_4.png', 1)
reference_image = cv2.imread('../resources/img_1.png', 1)

# Calculate the histograms of the input and reference images
hist_input = cv2.calcHist([input_image], [0], None, [256], [0, 256])
hist_reference = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

# Calculate the cumulative distribution functions (CDF) of the histograms
cdf_input = hist_input.cumsum()
cdf_reference = hist_reference.cumsum()

# Normalize the CDFs to the range [0, 255]
cdf_input_normalized = (cdf_input / cdf_input.max()) * 255
cdf_reference_normalized = (cdf_reference / cdf_reference.max()) * 255

# Create a mapping table for pixel value transformation
mapping_table = np.interp(cdf_input_normalized, cdf_reference_normalized, range(256))

# Apply the mapping table to the input image
output_image = cv2.LUT(input_image, mapping_table.astype(np.uint8))

# Display the input, reference, and output images
cv2.imshow('Input Image', input_image)
cv2.imshow('Reference Image', reference_image)
cv2.imshow('Specified Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the specified image
cv2.imwrite('../resources/specified_image.jpg', output_image)
