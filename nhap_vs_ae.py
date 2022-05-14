import cv2
import numpy as np
from matplotlib import pyplot as plt


#reading the image
img = cv2.imread("F:/CODE/DB1_B/101_2.tif",0)
img=cv2.bitwise_not(img)
ret, img=cv2.threshold(img,244,255,0)
cv2.imshow('img',img)
mG= [[1,1,0],
     [1,0,1],
     [0,0,1]]
#print(np.size(mG))
a = cv2.countNonZero(mG)
print(a)
cv2.waitKey(0)
cv2.destroyAllWindows()