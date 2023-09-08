import cv2
import numpy as np

img = cv2.imread("../resources/img.png",0)
gamma = [0.05, 0.5, 1, 1.5, 100]
for p in gamma:
    gammaCorrected = np.array(255*(img/255)**p, dtype=np.uint8)
    cv2.imshow(str(p)+' corrected', gammaCorrected)
cv2.waitKey(0)
cv2.destroyAllWindows()
