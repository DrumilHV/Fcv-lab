import cv2
import numpy as np

image = cv2.imread("../resources/img.png", 0)
cv2.imshow("original Image", image)
c = 255 / (np.log(1 + np.max(image)))
logTransform = c * np.log(1 + image)

logTransform = np.array(logTransform, dtype=np.uint8)
cv2.imshow("Trasformed Image", logTransform)

cv2.waitKey(0)
cv2.destroyAllWindows()
