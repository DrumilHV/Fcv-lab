# import cv2
# import numpy as np
#
# # loading the input and ref image
# inputImg = cv2.imread("../resources/img_4.png")
# refImg = cv2.imread("../resources/img_1.png")
#
# # Resize the images (optional)
# inputImg = cv2.resize(inputImg, (400, 400))
# refImg = cv2.resize(refImg, (400, 400))
#
# # calculating Histogram
# input_hist = cv2.calcHist([inputImg], [0], None, [256], [0, 256])
# ref_hist = cv2.calcHist([refImg], [0], None, [256], [0, 256])
#
# # calculating the cumulative Frequency
# input_cum = input_hist.cumsum()
# ref_cum = ref_hist.cumsum()
#
# # normalizing
# norm_inp = (input_cum / input_cum.max())*255
# norm_ref = (ref_cum / ref_cum.max())*255
#
# # mapping Table
# mapping_table = np.interp(norm_inp, norm_ref, range(256))
#
# # Mapping
# output_img = cv2.LUT(inputImg, mapping_table.astype(np.uint8))
#
# final_image = np.hstack((inputImg, refImg, output_img))
#
# cv2.imshow("Input vs ref vs spesifide", final_image)
# # cv2.imshow("input", inputImg)
# # cv2.imshow("ref" ,refImg)
# # cv2.imshow("specifide", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# img = cv2.imread('../resources/specified_image.jpg')
#
# bluredImg = cv2.GaussianBlur(img,(5,5),3)
#
# unshparpImg = cv2.addWeighted(img,2.5,bluredImg,-1.5,0)
#
# cv2.imshow('Original Image', img)
# cv2.imshow('unshparpImg', unshparpImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Load the input image
# input_img = cv2.imread('../resources/img_6.png', cv2.IMREAD_GRAYSCALE)
#
# # Apply Canny edge detection
# edges = cv2.Canny(input_img, threshold1=30, threshold2=100)  # Adjust thresholds as needed
# cv2.imshow("edges",edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Find contours in the binary image
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Create an empty mask to draw the segmented region
# segmented_mask = np.zeros_like(input_img)
#
# # Draw the contours on the mask
# cv2.drawContours(segmented_mask, contours, -1, (255), thickness=cv2.FILLED)
#
# # Apply the mask to the original image
# segmented_result = cv2.bitwise_and(input_img, input_img, mask=segmented_mask)
#
# # Display the segmented result
# cv2.imshow('Segmented Result', segmented_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv.imread('../resources/img_5.png')
#
# # height, width = image.shape[:2]
# # pixels = image.reshape((-1, 3))
# # pixels = np.float32(pixels)
# #
# # num_clusters = 3
# #
# # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# # _, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# # centers = np.uint8(centers)
# # segmented_image = centers[labels.flatten()]
# # segmented_image = segmented_image.reshape(image.shape)
# # # segmented_image = cv.resize('segmented_image',(0,0), fx=0.5, fy=0.5)
#
# # stacked_img = np.hstack((image, segmented_image))
# # cv.imshow('Original - Segmented', stacked_img)
#
# image = cv.medianBlur(image,7)
# image = cv.GaussianBlur(image, (11,11), 0)
# ret, thresh = cv.threshold(image, 120, 255, cv.THRESH_BINARY)
#
# # kernel = np.ones((3, 3), np.uint8)
# # closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,kernel, iterations = 15)
# # bg = cv.dilate(closing, kernel, iterations = 1)
# # dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
# # reta, fg = cv.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
# # cv.imshow('image', fg)
# # plt.figure(figsize=(8,8))
# # plt.imshow(fg,cmap="gray")
# # plt.axis('off')
# # plt.title("Segmented Image")
# # plt.show()
#
# lower_color = np.array([70, 90, 90])
# upper_color = np.array([255, 255, 255])
#
# mask = cv.inRange(thresh, lower_color, upper_color)
#
# seg = cv.bitwise_and(thresh, thresh, mask=mask)
#
# cv.imshow('Original Image', image)
# cv.imshow('Segmented Image', thresh)
# cv.imshow('try', seg)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Load the input image
# input_img = cv2.imread('../resources/specified_image.jpg')
#
# # Convert the image to grayscale
# gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#
# # Apply a Laplacian filter
# laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
#
# # Convert the result to an absolute value
# laplacian_abs = np.absolute(laplacian)
# laplacian_abs = np.uint8(laplacian_abs)
#
# # Display the original and Laplacian-filtered images
# cv2.imshow('Original Image', gray_img)
# cv2.imshow('Laplacian Filtered Image', laplacian_abs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
# # Load the input image
# input_img = cv2.imread('../resources/img_4.png', cv2.IMREAD_GRAYSCALE)
#
# # Apply Canny edge detection
# edges = cv2.Canny(input_img, threshold1=30, threshold2=100)  # Adjust thresholds as needed
#
# # Find contours in the binary image
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Create an empty mask to draw the segmented region
# segmented_mask = np.zeros_like(input_img)
#
# # Draw the contours on the mask
# cv2.drawContours(segmented_mask, contours, -1, (255), thickness=cv2.FILLED)
#
# # Apply the mask to the original image
# segmented_result = cv2.bitwise_and(input_img, input_img, mask=segmented_mask)
#
# # Display the segmented result
# cv2.imshow('Segmented Result', segmented_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# image = cv2.imread('../resources/img_4.png', cv2.COLOR_BGR2GRAY)
#
# edges = cv2.Canny(image, threshold1=30, threshold2=120)
#
# counters , _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# segmentMask  = np.zeros_like(image)
#
# cv2.drawContours(segmentMask, counters, -1, color=[255,0,0], thickness=cv2.FILLED)
#
# segRes = cv2.bitwise_and(image,image, segmentMask)
#
# cv2.imshow("segmentation ", segRes)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

